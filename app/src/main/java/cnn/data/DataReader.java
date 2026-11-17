package cnn.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import numj.core.NumJArray;

public class DataReader {

    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path) {

        List<Image> images = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {

            String line;

            while((line = reader.readLine()) != null) {
                
                String[] lineItems = line.split(",");

                NumJArray data = NumJArray.zeros(rows, cols);
                int label = Integer.parseInt(lineItems[0]);

                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        data.set(Integer.parseInt(lineItems[i * cols + j + 1]), i, j);
                    }
                }

                images.add(new Image(data, label));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return images;
    }

}
