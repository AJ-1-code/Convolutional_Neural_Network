package cnn.network;

import java.util.ArrayList;
import java.util.List;

import cnn.layers.ConvolutionLayer;
import cnn.layers.FullyConnectedLayer;
import cnn.layers.Layer;
import cnn.layers.MaxPoolLayer;

public class NetworkBuilder {

    private final int inputRows;
    private final int inputCols;
    private final float scaleFactor;
    private final List<Layer> layers;

    public NetworkBuilder(int inputRows, int inputCols, float scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
        this.layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, float learningRate, long seed) {
        layers.add(new ConvolutionLayer(filterSize, stepSize, currentLength(), currentRows(), currentCols(), seed, numFilters, learningRate));
    }

    public void addMaxPoolLayer(int windowSize, int stepSize) {
        layers.add(new MaxPoolLayer(stepSize, windowSize, currentLength(), currentRows(), currentCols()));
    }

    public void addFullyConnectedLayer(int outLength, float learningRate, long seed, boolean activate) {
        layers.add(new FullyConnectedLayer(currentElements(), outLength, seed, learningRate, activate));
    }

    public void addFullyConnectedLayer(int outLength, float learningRate, long seed) {
        addFullyConnectedLayer(outLength, learningRate, seed, true);
    }

    public NeuralNetwork build() {
        return new NeuralNetwork(layers, scaleFactor);
    }

    private int currentLength() {
        return layers.isEmpty() ? 1 : lastLayer().getOutputLength();
    }

    private int currentRows() {
        return layers.isEmpty() ? inputRows : lastLayer().getOutputRows();
    }

    private int currentCols() {
        return layers.isEmpty() ? inputCols : lastLayer().getOutputColumns();
    }

    private int currentElements() {
        return layers.isEmpty() ? inputRows * inputCols : lastLayer().getOutputElements();
    }

    private Layer lastLayer() {
        return layers.get(layers.size() - 1);
    }
}