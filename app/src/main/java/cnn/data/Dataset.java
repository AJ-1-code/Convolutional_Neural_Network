package cnn.data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import numj.core.NumJArray;

public class Dataset {

    public static final int SIZE = 64;

    private final List<String> classes = new ArrayList<>();
    private final List<float[]> drawings = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();

    public List<String> classNames() {
        return Collections.unmodifiableList(classes);
    }

    public int classCount() {
        return classes.size();
    }

    public int sampleCount() {
        return drawings.size();
    }

    public int sampleCount(int classIndex) {
        int count = 0;

        for (int label : labels) {
            if (label == classIndex) {
                count++;
            }
        }

        return count;
    }

    public void addClass(String name) {
        classes.add(name);
    }

    public void removeClass(int index) {
        for (int i = drawings.size() - 1; i >= 0; i--) {
            int label = labels.get(i);

            if (label == index) {
                drawings.remove(i);
                labels.remove(i);
            } else if (label > index) {
                labels.set(i, label - 1);
            }
        }

        classes.remove(index);
    }

    public void addSample(float[] drawing, int classIndex) {
        drawings.add(drawing.clone());
        labels.add(classIndex);
    }

    public List<Image> toImages(Random random, int maxShift) {
        List<Image> images = new ArrayList<>(drawings.size());

        for (int i = 0; i < drawings.size(); i++) {
            float[] pixels = drawings.get(i);

            if (maxShift > 0) {
                int dx = random.nextInt(2 * maxShift + 1) - maxShift;
                int dy = random.nextInt(2 * maxShift + 1) - maxShift;
                pixels = shift(pixels, dx, dy);
            }

            images.add(toImage(pixels, labels.get(i)));
        }

        return images;
    }

    public static Image toImage(float[] pixels, int label) {
        return new Image(new NumJArray(pixels, new int[]{SIZE, SIZE}), label);
    }

    public void save(Path path) {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(classes.size());

            for (String name : classes) {
                out.writeUTF(name);
            }

            out.writeInt(drawings.size());

            for (int i = 0; i < drawings.size(); i++) {
                out.writeInt(labels.get(i));

                for (float value : drawings.get(i)) {
                    out.writeFloat(value);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to save dataset: " + path, e);
        }
    }

    public static Dataset load(Path path) {
        Dataset dataset = new Dataset();

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int classCount = in.readInt();

            for (int i = 0; i < classCount; i++) {
                dataset.classes.add(in.readUTF());
            }

            int sampleCount = in.readInt();

            for (int i = 0; i < sampleCount; i++) {
                dataset.labels.add(in.readInt());
                float[] pixels = new float[SIZE * SIZE];

                for (int p = 0; p < pixels.length; p++) {
                    pixels[p] = in.readFloat();
                }

                dataset.drawings.add(pixels);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load dataset: " + path, e);
        }

        return dataset;
    }

    private static float[] shift(float[] source, int dx, int dy) {
        float[] result = new float[source.length];

        for (int row = 0; row < SIZE; row++) {
            int sourceRow = row - dy;

            if (sourceRow < 0 || sourceRow >= SIZE) {
                continue;
            }

            for (int col = 0; col < SIZE; col++) {
                int sourceCol = col - dx;

                if (sourceCol >= 0 && sourceCol < SIZE) {
                    result[row * SIZE + col] = source[sourceRow * SIZE + sourceCol];
                }
            }
        }

        return result;
    }
}