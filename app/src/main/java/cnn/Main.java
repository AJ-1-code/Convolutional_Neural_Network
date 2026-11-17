
package cnn;

import java.io.File;
import java.util.Collections;
import java.util.Random;
import java.util.List;

import cnn.data.DataReader;
import cnn.data.Image;
import cnn.network.BenchmarkMemory;
import cnn.network.NetworkBuilder;
import cnn.network.NeuralNetwork;

public class Main {

    private static final long SEED = 123;
    private static final float LEARNING_RATE = 0.03f;
    private static final int EPOCHS = 5;
    private static final int TRAIN_BATCH_SIZE = 64;
    private static final int TEST_BATCH_SIZE = 256;
    private static final String MODEL_DIRECTORY = "model";

    public static void main(String[] args) {

        System.out.println("");
        System.out.println("          NUMJ CNN BENCHMARK");
        System.out.println("");
        System.out.println("\nLoading datasets...");

        long dataStart = System.nanoTime();

        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");

        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");

        double dataSeconds = secondsSince(dataStart);

        System.out.println("Training images: " + imagesTrain.size());
        System.out.println("Testing images:  " + imagesTest.size());

        System.out.printf("Data loading time: %.2fs%n", dataSeconds);

        System.out.println("\nBuilding network...");

        Random random = new Random(SEED);
        NeuralNetwork net = buildNetwork();

        File modelFolder = new File(MODEL_DIRECTORY);

        if (modelFolder.exists() && new File(modelFolder, "layer0_0.nj").exists()) {
            System.out.println("Loading existing model...");
            net.load(MODEL_DIRECTORY);

        } else {
            System.out.println("No saved model found. Starting from scratch.");
        }

        System.out.println("\nTesting before training...");

        long initialTestStart = System.nanoTime();

        float initialAccuracy = net.test(imagesTest, TEST_BATCH_SIZE);
        double initialTestSeconds = secondsSince(initialTestStart);

        System.out.printf("Initial accuracy: %.2f%%%n", initialAccuracy * 100);
        System.out.printf("Initial test time: %.2fs%n", initialTestSeconds);

        double totalTrainingSeconds = 0;
        double totalTestingSeconds = 0;

        float finalAccuracy = initialAccuracy;

        long totalStart = System.nanoTime();

        BenchmarkMemory memory = new BenchmarkMemory();
        memory.start();

        System.out.println("\n");
        System.out.println("              TRAINING");
        System.out.println("");

        for (int epoch = 0; epoch < EPOCHS; epoch++) {

            System.out.println("\nEpoch " + (epoch + 1) + "/" + EPOCHS);

            Collections.shuffle(imagesTrain, random);

            int total = imagesTrain.size();
            long trainStart = System.nanoTime();

            net.train(imagesTrain, TRAIN_BATCH_SIZE, done -> printProgress(done, total));

            double trainSeconds = secondsSince(trainStart);
            totalTrainingSeconds += trainSeconds;

            double trainThroughput = imagesTrain.size() / trainSeconds;

            System.out.println();
            long testStart = System.nanoTime();

            finalAccuracy = net.test(imagesTest, TEST_BATCH_SIZE);

            double testSeconds = secondsSince(testStart);
            totalTestingSeconds += testSeconds;

            double testThroughput = imagesTest.size() / testSeconds;

            System.out.printf("  Accuracy:          %.2f%%%n", finalAccuracy * 100);

            System.out.printf("  Training time:     %.2fs%n", trainSeconds);

            System.out.printf("  Training speed:    %.0f images/sec%n", trainThroughput);

            System.out.printf("  Testing time:      %.2fs%n", testSeconds);

            System.out.printf("  Testing speed:     %.0f images/sec%n", testThroughput);
        }

        double totalSeconds = secondsSince(totalStart);
        memory.stop();

        System.out.println("\n");
        System.out.println("           BENCHMARK RESULTS");
        System.out.println("");

        System.out.printf("Initial accuracy:       %.2f%%%n", initialAccuracy * 100);

        System.out.printf("Final accuracy:         %.2f%%%n", finalAccuracy * 100);

        System.out.printf("Total training time:    %.2fs%n", totalTrainingSeconds);

        System.out.printf("Average epoch time:     %.2fs%n", totalTrainingSeconds / EPOCHS);

        System.out.printf("Total testing time:     %.2fs%n", totalTestingSeconds);

        System.out.printf("Total training images:  %d%n", (long) imagesTrain.size() * EPOCHS);

        System.out.printf("Average training speed: %.0f images/sec%n", (imagesTrain.size() * (double) EPOCHS) / totalTrainingSeconds);

        System.out.printf("Average testing speed:  %.0f images/sec%n", (imagesTest.size() * (double) EPOCHS) / totalTestingSeconds);

        System.out.printf("Total benchmark time:   %.2fs%n", totalSeconds);

        System.out.printf("Start memory:           %.2f MB%n", memory.getStartMB());

        System.out.printf("Peak memory:            %.2f MB%n", memory.getPeakMB());

        System.out.printf("End memory:             %.2f MB%n", memory.getEndMB());

        net.save(MODEL_DIRECTORY);

        System.out.println("\nSaved model to " + MODEL_DIRECTORY);
        System.out.println("");
    }

    private static NeuralNetwork buildNetwork() {

        NetworkBuilder builder = new NetworkBuilder(28, 28, 255.0f);

        builder.addConvolutionLayer(8, 5, 1, LEARNING_RATE, SEED);

        builder.addMaxPoolLayer(3, 2);

        builder.addFullyConnectedLayer(10, LEARNING_RATE, SEED + 1, false);

        return builder.build();
    }

    private static double secondsSince(long startNanos) {

        return (System.nanoTime() - startNanos) / 1_000_000_000.0;
    }

    private static void printProgress(int done, int total) {
        System.out.printf("\r  Training: %d / %d (%.1f%%)", done,total, 100.0 * done / total);
    }
}
