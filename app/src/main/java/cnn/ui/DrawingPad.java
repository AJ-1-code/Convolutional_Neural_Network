package cnn.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import cnn.data.Dataset;

public class DrawingPad extends JPanel {

    private static final int SIZE = Dataset.SIZE;
    private static final int TILE_SIZE = 10;
    private static final int BRUSH_RADIUS = 2;
    private static final float INK_THRESHOLD = 0.05f;

    private final float[][] pixels = new float[SIZE][SIZE];

    private int previousCol = -1;
    private int previousRow = -1;

    public DrawingPad() {
        setPreferredSize(new Dimension(SIZE * TILE_SIZE, SIZE * TILE_SIZE));

        MouseAdapter mouse = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent event) {
                if (SwingUtilities.isRightMouseButton(event)) {
                    clear();
                    return;
                }

                resetStroke();
                draw(event);
            }

            @Override
            public void mouseDragged(MouseEvent event) {
                draw(event);
            }

            @Override
            public void mouseReleased(MouseEvent event) {
                resetStroke();
            }
        };

        addMouseListener(mouse);
        addMouseMotionListener(mouse);
    }

    public float[] getInput() {
        int minCol = SIZE;
        int maxCol = -1;
        int minRow = SIZE;
        int maxRow = -1;

        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (pixels[row][col] > INK_THRESHOLD) {
                    minCol = Math.min(minCol, col);
                    maxCol = Math.max(maxCol, col);
                    minRow = Math.min(minRow, row);
                    maxRow = Math.max(maxRow, row);
                }
            }
        }

        float[] input = new float[SIZE * SIZE];

        if (maxCol == -1) {
            return input;
        }

        int colShift = Math.round((SIZE - 1 - minCol - maxCol) / 2.0f);
        int rowShift = Math.round((SIZE - 1 - minRow - maxRow) / 2.0f);

        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                int sourceRow = row - rowShift;
                int sourceCol = col - colShift;

                if (sourceRow >= 0 && sourceRow < SIZE && sourceCol >= 0 && sourceCol < SIZE) {
                    input[row * SIZE + col] = pixels[sourceRow][sourceCol];
                }
            }
        }

        return input;
    }

    public boolean isBlank() {
        for (float[] row : pixels) {
            for (float pixel : row) {
                if (pixel > INK_THRESHOLD) {
                    return false;
                }
            }
        }

        return true;
    }

    public void clear() {
        for (float[] row : pixels) {
            java.util.Arrays.fill(row, 0f);
        }

        repaint();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);

        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                int shade = 255 - Math.round(255 * pixels[row][col]);
                graphics.setColor(new Color(shade, shade, shade));
                graphics.fillRect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            }
        }

        graphics.setColor(new Color(0, 0, 0, 40));

        for (int i = 0; i <= SIZE; i++) {
            graphics.drawLine(0, i * TILE_SIZE, SIZE * TILE_SIZE, i * TILE_SIZE);
            graphics.drawLine(i * TILE_SIZE, 0, i * TILE_SIZE, SIZE * TILE_SIZE);
        }
    }

    private void draw(MouseEvent event) {
        if (!SwingUtilities.isLeftMouseButton(event)) {
            return;
        }

        int col = event.getX() / TILE_SIZE;
        int row = event.getY() / TILE_SIZE;

        if (col < 0 || col >= SIZE || row < 0 || row >= SIZE) {
            return;
        }

        if (previousCol >= 0) {
            drawLine(previousCol, previousRow, col, row);
        } else {
            paintBrush(col, row);
        }

        previousCol = col;
        previousRow = row;
        repaint();
    }

    private void drawLine(int startCol, int startRow, int endCol, int endRow) {
        int steps = Math.max(Math.abs(endCol - startCol), Math.abs(endRow - startRow));

        for (int step = 0; step <= steps; step++) {
            float t = steps == 0 ? 0f : (float) step / steps;
            paintBrush(Math.round(startCol + (endCol - startCol) * t), Math.round(startRow + (endRow - startRow) * t));
        }
    }

    private void paintBrush(int col, int row) {
        for (int y = row - BRUSH_RADIUS; y <= row + BRUSH_RADIUS; y++) {
            for (int x = col - BRUSH_RADIUS; x <= col + BRUSH_RADIUS; x++) {
                if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
                    continue;
                }

                double distance = Math.hypot(x - col, y - row);

                if (distance <= BRUSH_RADIUS) {
                    float falloff = (float) (1.0 - distance / (BRUSH_RADIUS + 1));
                    pixels[y][x] = Math.max(pixels[y][x], falloff);
                }
            }
        }
    }

    private void resetStroke() {
        previousCol = -1;
        previousRow = -1;
    }
}