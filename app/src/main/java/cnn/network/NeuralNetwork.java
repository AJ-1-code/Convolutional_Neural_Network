package cnn.network;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;
import java.util.function.IntConsumer;
import java.nio.file.Path;
import java.util.Arrays;

import cnn.data.Image;
import cnn.layers.Layer;
import numj.core.NumJArray;

public class NeuralNetwork {

    private final List<Layer> layers;
    private final float scaleFactor;

    public NeuralNetwork(List<Layer> layers, float scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    public int guess(Image image) {
        NumJArray output = forward(Collections.singletonList(image), 0, 1);
        return predictions(output)[0];
    }

    public float test(List<Image> images, int batchSize) {
        int correct = 0;

        for (int start = 0; start < images.size(); start += batchSize) {
            int count = Math.min(batchSize, images.size() - start);
            int[] predicted = predictions(forward(images, start, count));

            for (int b = 0; b < count; b++) {
                if (predicted[b] == images.get(start + b).getLabel()) {
                    correct++;
                }
            }
        }

        return (float) correct / images.size();
    }

    private int[] predictions(NumJArray output) {
        float[] indices = output.argMaxAxis(1).readOnlyData();
        int[] result = new int[indices.length];

        for (int i = 0; i < indices.length; i++) {
            result[i] = (int) indices[i];
        }

        return result;
    }

    public float train(List<Image> images, int batchSize) {
        return train(images, batchSize, null);
    }
    
    public float train(List<Image> images, int batchSize, IntConsumer onProgress) {
        int total = images.size();
        float totalLoss = 0f;
    
        for (int start = 0; start < total; start += batchSize) {
            int count = Math.min(batchSize, total - start);
            NumJArray output = forward(images, start, count);
            NumJArray expected = oneHotLabels(images, start, count, output.shape[1]);
        
            totalLoss += expected.smce(output) * count;
            layers.get(layers.size() - 1).backward(expected.smceD(output));
        
            if (onProgress != null) {
                onProgress.accept(start + count);
            }
        }
    
        return totalLoss / total;
    }
    
    public float[] predict(Image image) {
        NumJArray output = forward(Collections.singletonList(image), 0, 1);
        return output.softmaxAxis(1).toContiguousData();
    }


    private void linkLayers() {
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            current.setNextLayer(next);
            next.setPrevLayer(current);
        }
    }

    public void save(String directory) {
        try {
            Files.createDirectories(Path.of(directory));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to create directory: " + directory, e);
        }

        for (int i = 0; i < layers.size(); i++) {
            NumJArray[] parameters = layers.get(i).parameters();

            for (int j = 0; j < parameters.length; j++) {
                parameters[j].save(parameterPath(directory, i, j));
            }
        }
    }

    public void load(String directory) {
        for (int i = 0; i < layers.size(); i++) {
            NumJArray[] parameters = layers.get(i).parameters();

            for (int j = 0; j < parameters.length; j++) {
                NumJArray loaded = NumJArray.load(parameterPath(directory, i, j));

                if (!Arrays.equals(loaded.shape, parameters[j].shape)) {
                    throw new IllegalArgumentException(
                            "Saved shape " + Arrays.toString(loaded.shape) + " does not match layer " + i + " shape " + Arrays.toString(parameters[j].shape));
                }

                parameters[j].assign(loaded);
            }
        }
    }

    private String parameterPath(String directory, int layer, int parameter) {
        return Path.of(directory, "layer" + layer + "_" + parameter + ".nj").toString();
    }

    private NumJArray forward(List<Image> images, int start, int count) {
        return layers.get(0).forward(buildBatch(images, start, count));
    }

    private NumJArray buildBatch(List<Image> images, int start, int count) {
        int rows = images.get(0).getData().shape[0];
        int cols = images.get(0).getData().shape[1];
        int size = rows * cols;
        float inverseScale = 1.0f / scaleFactor;

        NumJArray batch = NumJArray.zeros(count, 1, rows, cols);
        float[] target = batch.data;

        for (int b = 0; b < count; b++) {
            float[] source = images.get(start + b).getData().readOnlyData();
            int base = b * size;

            for (int p = 0; p < size; p++) {
                target[base + p] = source[p] * inverseScale;
            }
        }

        return batch;
    }

    private NumJArray oneHotLabels(List<Image> images, int start, int count, int numClasses) {
        NumJArray expected = NumJArray.zeros(count, numClasses);

        for (int b = 0; b < count; b++) {
            expected.data[b * numClasses + images.get(start + b).getLabel()] = 1f;
        }

        return expected;
    }
}