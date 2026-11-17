package cnn.layers;

import java.util.Random;

import numj.core.NumJArray;

public class FullyConnectedLayer extends Layer {

    private static final float LEAK = 0.01f;

    private final int outLength;
    private final float learningRate;
    private final boolean activate;

    private final NumJArray weights;

    private NumJArray lastX;
    private NumJArray lastZ;

    public FullyConnectedLayer(int inLength, int outLength, long seed, float learningRate, boolean activate) {
        this.outLength = outLength;
        this.learningRate = learningRate;
        this.activate = activate;
        this.weights = heInit(new Random(seed), inLength, inLength, outLength);
    }

    @Override
    public NumJArray forward(NumJArray input) {
        if (input.ndim() > 2) {
            input = reshapeToVector(input);
        }

        lastX = input;
        lastZ = input.matMul(weights);

        NumJArray output = activate ? lastZ.leakyRelu(LEAK) : lastZ;
        return forwardNext(output);
    }

    @Override
    public NumJArray backward(NumJArray dldo) {
        NumJArray dldz = activate ? dldo.muli(lastZ.leakyReluD(LEAK)) : dldo;
        NumJArray dldx = dldz.matMul(weights.transpose());
        NumJArray dldw = lastX.transpose().matMul(dldz);

        weights.subi(dldw.muli(learningRate));
        return backwardPrev(dldx);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputColumns() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outLength;
    }

    @Override
    public NumJArray[] parameters() {
        return new NumJArray[]{weights};
    }
}