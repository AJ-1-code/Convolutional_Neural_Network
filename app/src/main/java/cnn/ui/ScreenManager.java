package cnn.ui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRootPane;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;

import cnn.data.Dataset;
import cnn.data.Image;
import cnn.network.NetworkBuilder;
import cnn.network.NeuralNetwork;

public class ScreenManager {

    private static final String DATASET_FILE = "dataset.bin";
    private static final String MODEL_DIRECTORY = "model";
    private static final String CLASSES_FILE = "classes.txt";
    private static final String[] DEFAULT_CLASSES = {"LEFT ARROW", "RIGHT ARROW", "UP ARROW"};

    private static final long SEED = 123;
    private static final float LEARNING_RATE = 0.03f;
    private static final int EPOCHS = 60;
    private static final int BATCH_SIZE = 8;
    private static final int MAX_SHIFT = 4;

    private final Random random = new Random(SEED);
    private final Dataset dataset;
    private final DrawingPad pad = new DrawingPad();
    private final DefaultListModel<String> listModel = new DefaultListModel<>();
    private final JList<String> classList = new JList<>(listModel);
    private final JLabel statusLabel = new JLabel();
    private final List<JButton> buttons = new ArrayList<>();

    private NeuralNetwork net;
    private boolean training;

    public ScreenManager() {
        dataset = loadDataset();
        net = loadModel();

        buildWindow();
        refreshClasses();

        setStatus(net == null ? "Select a class, draw, and press Enter to add samples." : "Loaded saved model. Draw and press G to guess.");
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ScreenManager::new);
    }

    private void buildWindow() {
        JFrame window = new JFrame("Shape Classifier");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setResizable(false);

        classList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        classList.setFocusable(false);

        JScrollPane listPane = new JScrollPane(classList);
        listPane.setPreferredSize(new Dimension(220, 200));

        JPanel buttonPanel = new JPanel(new GridLayout(0, 1, 4, 4));
        buttonPanel.add(createButton("Add class", this::addClass));
        buttonPanel.add(createButton("Remove class", this::removeClass));
        buttonPanel.add(createButton("Add sample (Enter)", this::addSample));
        buttonPanel.add(createButton("Guess (G)", this::guess));
        buttonPanel.add(createButton("Clear (C)", pad::clear));
        buttonPanel.add(createButton("Train", this::train));

        JPanel side = new JPanel(new BorderLayout(0, 8));
        side.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        side.add(new JLabel("Classes (samples)"), BorderLayout.NORTH);
        side.add(listPane, BorderLayout.CENTER);
        side.add(buttonPanel, BorderLayout.SOUTH);

        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 8, 6, 8));

        window.add(pad, BorderLayout.CENTER);
        window.add(side, BorderLayout.EAST);
        window.add(statusLabel, BorderLayout.SOUTH);

        JRootPane root = window.getRootPane();
        bindKey(root, KeyEvent.VK_ENTER, "add-sample", this::addSample);
        bindKey(root, KeyEvent.VK_G, "guess", this::guess);
        bindKey(root, KeyEvent.VK_C, "clear", pad::clear);

        window.pack();
        window.setLocationRelativeTo(null);
        window.setVisible(true);
    }

    private JButton createButton(String text, Runnable action) {
        JButton button = new JButton(text);
        button.setFocusable(false);
        button.addActionListener(event -> action.run());
        buttons.add(button);
        return button;
    }

    private void bindKey(JRootPane root, int keyCode, String name, Runnable action) {
        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(keyCode, 0), name);
        root.getActionMap().put(name, new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                action.run();
            }
        });
    }

    private void addClass() {
        if (training) {
            return;
        }

        String name = JOptionPane.showInputDialog(classList, "Class name:", "Add class", JOptionPane.PLAIN_MESSAGE);

        if (name == null || name.isBlank()) {
            return;
        }

        name = name.trim().toUpperCase();

        if (dataset.classNames().contains(name)) {
            setStatus("Class already exists: " + name);
            return;
        }

        dataset.addClass(name);
        dataset.save(Path.of(DATASET_FILE));
        net = null;

        refreshClasses();
        classList.setSelectedIndex(dataset.classCount() - 1);
        setStatus("Added " + name + ". Model reset, train again after adding samples.");
    }

    private void removeClass() {
        if (training) {
            return;
        }

        int index = classList.getSelectedIndex();

        if (index < 0) {
            setStatus("Select a class to remove.");
            return;
        }

        String name = dataset.classNames().get(index);
        int answer = JOptionPane.showConfirmDialog(classList, "Remove " + name + " and its " + dataset.sampleCount(index) + " samples?", "Remove class", JOptionPane.YES_NO_OPTION);

        if (answer != JOptionPane.YES_OPTION) {
            return;
        }

        dataset.removeClass(index);
        dataset.save(Path.of(DATASET_FILE));
        net = null;

        classList.clearSelection();
        refreshClasses();
        setStatus("Removed " + name + ". Model reset, train again.");
    }

    private void addSample() {
        if (training) {
            return;
        }

        int index = classList.getSelectedIndex();

        if (index < 0) {
            setStatus("Select a class first.");
            return;
        }

        if (pad.isBlank()) {
            setStatus("Draw something first.");
            return;
        }

        dataset.addSample(pad.getInput(), index);
        dataset.save(Path.of(DATASET_FILE));
        pad.clear();

        refreshClasses();
        setStatus("Added to " + dataset.classNames().get(index) + " (" + dataset.sampleCount(index) + " samples).");
    }

    private void guess() {
        if (training) {
            return;
        }

        if (net == null) {
            setStatus("No trained model. Train first.");
            return;
        }

        if (pad.isBlank()) {
            setStatus("Draw something first.");
            return;
        }

        float[] probabilities = net.predict(Dataset.toImage(pad.getInput(), 0));
        int best = 0;

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }

        setStatus(String.format("Prediction: %s (%.0f%% confident)", dataset.classNames().get(best), probabilities[best] * 100));
    }

    private void train() {
        if (training) {
            return;
        }

        if (dataset.classCount() < 2) {
            setStatus("Need at least two classes.");
            return;
        }

        for (int i = 0; i < dataset.classCount(); i++) {
            if (dataset.sampleCount(i) == 0) {
                setStatus(dataset.classNames().get(i) + " has no samples.");
                return;
            }
        }

        NeuralNetwork target = net != null ? net : buildNetwork(dataset.classCount());
        setTraining(true);

        new SwingWorker<Float, String>() {
            @Override
            protected Float doInBackground() {
                for (int epoch = 1; epoch <= EPOCHS; epoch++) {
                    List<Image> images = dataset.toImages(random, MAX_SHIFT);
                    Collections.shuffle(images, random);

                    float loss = target.train(images, BATCH_SIZE);
                    publish(String.format("Training: epoch %d/%d, loss %.4f", epoch, EPOCHS, loss));
                }

                saveModel(target);
                return target.test(dataset.toImages(random, 0), BATCH_SIZE);
            }

            @Override
            protected void process(List<String> messages) {
                setStatus(messages.get(messages.size() - 1));
            }

            @Override
            protected void done() {
                setTraining(false);

                try {
                    float accuracy = get();
                    net = target;
                    setStatus(String.format("Done. Training accuracy %.1f%%. Model saved.", accuracy * 100));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    setStatus("Training interrupted.");
                } catch (ExecutionException e) {
                    e.printStackTrace();
                    setStatus("Training failed: " + e.getCause());
                }
            }
        }.execute();
    }

    private void setTraining(boolean value) {
        training = value;

        for (JButton button : buttons) {
            button.setEnabled(!value);
        }
    }

    private void refreshClasses() {
        int selected = classList.getSelectedIndex();
        List<String> names = dataset.classNames();

        listModel.clear();

        for (int i = 0; i < names.size(); i++) {
            listModel.addElement(names.get(i) + " (" + dataset.sampleCount(i) + ")");
        }

        if (selected >= 0 && selected < listModel.size()) {
            classList.setSelectedIndex(selected);
        }
    }

    private void setStatus(String text) {
        statusLabel.setText(text);
    }

    private void saveModel(NeuralNetwork network) {
        network.save(MODEL_DIRECTORY);

        try {
            Files.write(Path.of(MODEL_DIRECTORY, CLASSES_FILE), dataset.classNames());
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to save class names", e);
        }
    }

    private NeuralNetwork loadModel() {
        Path classesPath = Path.of(MODEL_DIRECTORY, CLASSES_FILE);

        try {
            if (!Files.exists(classesPath) || !Files.readAllLines(classesPath).equals(dataset.classNames())) {
                return null;
            }

            NeuralNetwork loaded = buildNetwork(dataset.classCount());
            loaded.load(MODEL_DIRECTORY);
            return loaded;
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    private static Dataset loadDataset() {
        Path path = Path.of(DATASET_FILE);

        if (Files.exists(path)) {
            return Dataset.load(path);
        }

        Dataset created = new Dataset();

        for (String name : DEFAULT_CLASSES) {
            created.addClass(name);
        }

        return created;
    }

    private static NeuralNetwork buildNetwork(int numClasses) {
        NetworkBuilder builder = new NetworkBuilder(Dataset.SIZE, Dataset.SIZE, 1.0f);
        builder.addConvolutionLayer(8, 5, 1, LEARNING_RATE, SEED);
        builder.addMaxPoolLayer(2, 2);
        builder.addConvolutionLayer(16, 5, 1, LEARNING_RATE, SEED + 1);
        builder.addMaxPoolLayer(2, 2);
        builder.addFullyConnectedLayer(numClasses, LEARNING_RATE, SEED + 2, false);
        return builder.build();
    }
}