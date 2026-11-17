package cnn.layers;

import java.util.Random;

import numj.core.NumJArray;

public class ConvolutionLayer extends Layer {

    private final int filterSize;
    private final int stepSize;
    private final int inLength;
    private final int inRows;
    private final int inCols;
    private final int numFilters;
    private final float learningRate;

    private final NumJArray filters;

    private NumJArray lastCols;
    private NumJArray lastActivation;  

    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters, float learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
        this.numFilters = numFilters;
        this.learningRate = learningRate;
        this.filters = heInit(new Random(seed), inLength * filterSize * filterSize, numFilters, inLength, filterSize, filterSize);
    }

    @Override
    public NumJArray forward(NumJArray input) {
        int batch = input.shape[0];
        lastCols = im2col(input, batch);

        NumJArray filterMatrix = filters.reshape(numFilters, patchSize());
        NumJArray output = filtersToBatch(filterMatrix.matMul(lastCols), batch).relu();

        lastActivation = output;
        return forwardNext(output);
    }

    @Override
    public NumJArray backward(NumJArray dldo) {
        int batch = dldo.shape[0];

        if (dldo.ndim() == 2) {
            dldo = reshapeToMatrix(dldo, numFilters, getOutputRows(), getOutputColumns());
        }

        dldo = dldo.copy().muli(lastActivation.reluD());

        NumJArray dY = batchToFilters(dldo, batch);
        NumJArray filterMatrix = filters.reshape(numFilters, patchSize());

        NumJArray dCols = filterMatrix.transpose().matMul(dY);
        NumJArray dldx = col2im(dCols, batch);
        NumJArray dW = dY.matMul(lastCols.transpose()).muli(learningRate);

        filters.subi(dW.reshape(numFilters, inLength, filterSize, filterSize));
        return backwardPrev(dldx);
    }

    private int patchSize() {
        return inLength * filterSize * filterSize;
    }

    private NumJArray im2col(NumJArray input, int batch) {
        int outR = getOutputRows();
        int outC = getOutputColumns();
        int positions = outR * outC;
        int columns = batch * positions;

        float[] in = input.data;
        int offset = input.offset;
        int batchStride = input.strides[0];
        int channelStride = input.strides[1];
        int rowStride = input.strides[2];
        int colStride = input.strides[3];

        float[] out = new float[patchSize() * columns];

        for (int c = 0; c < inLength; c++) {
            for (int x = 0; x < filterSize; x++) {
                for (int y = 0; y < filterSize; y++) {
                    int rowBase = ((c * filterSize + x) * filterSize + y) * columns;

                    for (int b = 0; b < batch; b++) {
                        int colBase = rowBase + b * positions;
                        int inBase = offset + b * batchStride + c * channelStride;

                        for (int i = 0; i < outR; i++) {
                            int inRow = inBase + (i * stepSize + x) * rowStride;
                            int o = colBase + i * outC;

                            for (int j = 0; j < outC; j++) {
                                out[o + j] = in[inRow + (j * stepSize + y) * colStride];
                            }
                        }
                    }
                }
            }
        }

        return new NumJArray(out, new int[]{patchSize(), columns}, new int[]{columns, 1}, 0);
    }

    private NumJArray col2im(NumJArray dCols, int batch) {
        int outR = getOutputRows();
        int outC = getOutputColumns();
        int positions = outR * outC;
        int columns = batch * positions;
        int channelSize = inRows * inCols;

        float[] gradient = dCols.data;
        float[] dx = new float[batch * inLength * channelSize];

        for (int c = 0; c < inLength; c++) {
            for (int x = 0; x < filterSize; x++) {
                for (int y = 0; y < filterSize; y++) {
                    int rowBase = ((c * filterSize + x) * filterSize + y) * columns;

                    for (int b = 0; b < batch; b++) {
                        int colBase = rowBase + b * positions;
                        int dxBase = (b * inLength + c) * channelSize;

                        for (int i = 0; i < outR; i++) {
                            int dxRow = dxBase + (i * stepSize + x) * inCols;
                            int o = colBase + i * outC;

                            for (int j = 0; j < outC; j++) {
                                dx[dxRow + j * stepSize + y] += gradient[o + j];
                            }
                        }
                    }
                }
            }
        }

        return new NumJArray(dx, new int[]{batch, inLength, inRows, inCols}, new int[]{inLength * channelSize, channelSize, inCols, 1},0);
    }

    private NumJArray filtersToBatch(NumJArray matrix, int batch) {
        int outR = getOutputRows();
        int outC = getOutputColumns();
        int positions = outR * outC;
        int columns = batch * positions;

        float[] source = matrix.data;
        float[] target = new float[batch * numFilters * positions];

        for (int f = 0; f < numFilters; f++) {
            for (int b = 0; b < batch; b++) {
                System.arraycopy(source, f * columns + b * positions, target, (b * numFilters + f) * positions, positions);
            }
        }

        return new NumJArray(target, new int[]{batch, numFilters, outR, outC},new int[]{numFilters * positions, positions, outC, 1},0);
    }

    private NumJArray batchToFilters(NumJArray tensor, int batch) {
        int positions = getOutputRows() * getOutputColumns();
        int columns = batch * positions;

        float[] source = tensor.data;
        float[] target = new float[numFilters * columns];

        for (int f = 0; f < numFilters; f++) {
            for (int b = 0; b < batch; b++) {
                System.arraycopy(source, (b * numFilters + f) * positions, target, f * columns + b * positions, positions);
            }
        }

        return new NumJArray(target, new int[]{numFilters, columns}, new int[]{columns, 1}, 0);
    }

    @Override
    public int getOutputLength() {
        return numFilters;
    }

    @Override
    public int getOutputRows() {
        return (inRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (inCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputColumns();
    }

    @Override
    public NumJArray[] parameters() {
        return new NumJArray[]{filters};
    }
}