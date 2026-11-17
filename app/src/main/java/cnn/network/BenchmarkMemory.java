package cnn.network;

import java.io.File;
import java.util.concurrent.atomic.AtomicLong;

public class BenchmarkMemory {

    private final AtomicLong peakHeap = new AtomicLong();

    private volatile boolean running;
    private Thread monitor;

    private long startHeap;
    private long endHeap;

    public void start() {

        startHeap = usedHeap();
        peakHeap.set(startHeap);

        running = true;

        monitor = new Thread(() -> {

            while (running) {
                long current = usedHeap();

                peakHeap.accumulateAndGet(current, Math::max);

                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    break;
                }
            }

        });

        monitor.setDaemon(true);
        monitor.start();
    }

    public void stop() {

        running = false;

        try {
            monitor.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        endHeap = usedHeap();

        peakHeap.accumulateAndGet(endHeap, Math::max);
    }

    private static long usedHeap() {

        Runtime runtime = Runtime.getRuntime();

        return runtime.totalMemory() - runtime.freeMemory();
    }

    public double getStartMB() {
        return toMB(startHeap);
    }

    public double getPeakMB() {
        return toMB(peakHeap.get());
    }

    public double getEndMB() {
        return toMB(endHeap);
    }

    public static double getFileSizeMB(File file) {

        if (!file.exists()) {
            return 0;
        }

        if (file.isFile()) {
            return toMB(file.length());
        }

        long size = 0;

        File[] files = file.listFiles();

        if (files != null) {
            for (File child : files) {
                size += getFileSizeBytes(child);
            }
        }

        return toMB(size);
    }

    private static long getFileSizeBytes(File file) {

        if (file.isFile()) {
            return file.length();
        }

        long size = 0;

        File[] files = file.listFiles();

        if (files != null) {
            for (File child : files) {
                size += getFileSizeBytes(child);
            }
        }

        return size;
    }

    private static double toMB(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }
}
