package cnn.data;

import numj.core.NumJArray;

public class Image {

    private NumJArray data;
    private int label;

    public Image(NumJArray data, int label) {
        this.data = data;
        this.label = label;
    }

    public NumJArray getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return data.toPrettyString();
    }
    

}
