package cnn.layers;

import numj.core.NumJArray;

public class MaxPoolLayer extends Layer {

    private final int stepSize;
    private final int windowSize;
    private final int inLength;
    private final int inRows;
    private final int inCols;

    private int[] argmax;

    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
    }

    @Override
    public NumJArray forward(NumJArray input) {
        int batch = input.shape[0];
        int outR = getOutputRows();
        int outC = getOutputColumns();
        int channelSize = inRows * inCols;

        float[] in = input.data;
        int offset = input.offset;
        int rowStride = input.strides[2];
        int colStride = input.strides[3];

        float[] out = new float[batch * inLength * outR * outC];
        argmax = new int[out.length];
        int o = 0;

        for (int b = 0; b < batch; b++) {
            for (int l = 0; l < inLength; l++) {
                int base = offset + b * input.strides[0] + l * input.strides[1];

                for (int i = 0; i < outR; i++) {
                    for (int j = 0; j < outC; j++) {
                        float max = Float.NEGATIVE_INFINITY;
                        int best = 0;

                        for (int x = 0; x < windowSize; x++) {
                            for (int y = 0; y < windowSize; y++) {
                                int r = i * stepSize + x;
                                int c = j * stepSize + y;
                                float value = in[base + r * rowStride + c * colStride];

                                if (value > max) {
                                    max = value;
                                    best = r * inCols + c;
                                }
                            }
                        }

                        out[o] = max;
                        argmax[o] = (b * inLength + l) * channelSize + best;
                        o++;
                    }
                }
            }
        }

        NumJArray output = new NumJArray(out,new int[]{batch, inLength, outR, outC}, new int[]{inLength * outR * outC, outR * outC, outC, 1},0);
        return forwardNext(output);
    }

    @Override
    public NumJArray backward(NumJArray dldo) {
        int batch = dldo.shape[0];

        if (dldo.ndim() == 2) {
            dldo = reshapeToMatrix(dldo, inLength, getOutputRows(), getOutputColumns());
        }

        float[] gradient = dldo.readOnlyData();
        float[] dx = new float[batch * inLength * inRows * inCols];

        for (int n = 0; n < gradient.length; n++) {
            dx[argmax[n]] += gradient[n];
        }

        NumJArray dldx = new NumJArray(dx, new int[]{batch, inLength, inRows, inCols},new int[]{inLength * inRows * inCols, inRows * inCols, inCols, 1},0);
        return backwardPrev(dldx);
    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (inCols - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength * getOutputRows() * getOutputColumns();
    }
}