package cnn.layers;

import java.util.Random;

import numj.core.NumJArray;

public abstract class Layer {

    protected Layer nextLayer;
    protected Layer prevLayer;

    public abstract NumJArray forward(NumJArray input);

    public abstract NumJArray backward(NumJArray dldo);

    public abstract int getOutputLength();

    public abstract int getOutputRows();

    public abstract int getOutputColumns();

    public abstract int getOutputElements();

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void setPrevLayer(Layer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public Layer getPrevLayer() {
        return prevLayer;
    }
    
    public NumJArray[] parameters() {
        return new NumJArray[0];
    }

    protected NumJArray forwardNext(NumJArray output) {
        if (nextLayer != null) {
            return nextLayer.forward(output);
        }
        return output;
    }

    protected NumJArray backwardPrev(NumJArray gradient) {
        if (prevLayer != null) {
            return prevLayer.backward(gradient);
        }
        return gradient;
    }

    protected NumJArray reshapeToVector(NumJArray input) {
        int batchSize = input.shape[0];
        int elementsPerSample = input.size() / batchSize;
        return input.reshape(new int[]{batchSize, elementsPerSample});
    }

    protected NumJArray reshapeToMatrix(NumJArray input, int channels, int rows, int cols) {
        int batchSize = input.shape[0];
        return input.reshape(new int[]{batchSize, channels, rows, cols});
    }

    protected static NumJArray heInit(Random random, int fanIn, int... shape) {
        NumJArray weights = NumJArray.zeros(shape);
        float[] data = weights.data;
        float std = (float) Math.sqrt(2.0 / fanIn);

        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (random.nextGaussian() * std);
        }
        return weights;
    }
}