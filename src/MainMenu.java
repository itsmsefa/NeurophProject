import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class MainMenu {
    // Store the best topologies found at startup
    private static int[] bestMomentumTopology = null;
    private static double bestMomentumMse = Double.MAX_VALUE;

    private static int[] bestNoMomentumTopology = null;
    private static double bestNoMomentumMse = Double.MAX_VALUE;

    public static void main(String[] args) {
        // Load train and test datasets
        DataSet trainData = DataLoader.loadDataSet("train.csv");
        DataSet testData = DataLoader.loadDataSet("test.csv");

        // Run the topology experiment at startup
        runTopologyExperiments(trainData, testData);

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1. Train and Test (With Momentum) - Uses best momentum topology");
            System.out.println("2. Train and Test (Without Momentum) - Uses best no-momentum topology");
            System.out.println("3. Epoch-by-Epoch Evaluation (Without Momentum)");
            System.out.println("4. Train and Single Test (With Momentum)");
            System.out.println("5. K-Fold Cross-Validation");
            System.out.println("0. Exit");
            System.out.print("Choose an option: ");
            int choice = scanner.nextInt();

            if (choice == 0) {
                System.out.println("Exiting program.");
                break;
            }

            switch (choice) {
                case 1 -> trainAndTest(trainData, testData, true);   // With momentum uses bestMomentumTopology
                case 2 -> trainAndTest(trainData, testData, false);  // Without momentum uses bestNoMomentumTopology
                case 3 -> epochByEpochEvaluation(trainData, testData);
                case 4 -> trainAndSingleTest(trainData, scanner);
                case 5 -> kFoldCrossValidation(trainData, scanner);
                default -> System.out.println("Invalid choice! Please select a valid option.");
            }
        }

        scanner.close();
    }

    private static void runTopologyExperiments(DataSet trainData, DataSet testData) {
        int[][] topologies = {
            {5},
            {10},
            {10, 10},
            {20},
            {20, 10},
            {10, 5},
            {15, 15},
            {5, 5, 5},
            {10, 10, 10},
            {20, 20}
        };

        XYSeries momentumFinalLossSeries = new XYSeries("Final Loss (With Momentum)");
        XYSeries noMomentumFinalLossSeries = new XYSeries("Final Loss (Without Momentum)");
        XYSeriesCollection momentumEpochLossDataset = new XYSeriesCollection();
        XYSeriesCollection noMomentumEpochLossDataset = new XYSeriesCollection();

        System.out.println("Evaluating topologies for With Momentum and Without Momentum...");

        for (int t = 0; t < topologies.length; t++) {
            int[] topology = topologies[t];
            System.out.println("Evaluating Topology " + topologyToString(topology));

            // Train with momentum
            NeuralNetwork<?> netMomentum = createNetwork(topology, 3, 1);
            List<Double> trainLossMomentum = trainNetworkWithLossTracking(netMomentum, trainData, 100, 0.1, 0.9);
            double testMseMomentum = MseCalculator.testNetwork(netMomentum, testData);

            // Add final test MSE to the final loss graph
            momentumFinalLossSeries.add(t + 1, testMseMomentum);

            // Add epoch-by-epoch training losses to the dataset
            XYSeries momentumEpochSeries = new XYSeries("Topology " + (t + 1) + " (With Momentum)");
            for (int epoch = 0; epoch < trainLossMomentum.size(); epoch++) {
                momentumEpochSeries.add(epoch + 1, trainLossMomentum.get(epoch));
            }
            momentumEpochLossDataset.addSeries(momentumEpochSeries);

            if (testMseMomentum < bestMomentumMse) {
                bestMomentumMse = testMseMomentum;
                bestMomentumTopology = topology;
            }

            // Train without momentum
            NeuralNetwork<?> netNoMomentum = createNetwork(topology, 3, 1);
            List<Double> trainLossNoMomentum = trainNetworkWithLossTracking(netNoMomentum, trainData, 100, 0.1, 0.0);
            double testMseNoMomentum = MseCalculator.testNetwork(netNoMomentum, testData);

            // Add final test MSE to the final loss graph
            noMomentumFinalLossSeries.add(t + 1, testMseNoMomentum);

            // Add epoch-by-epoch training losses to the dataset
            XYSeries noMomentumEpochSeries = new XYSeries("Topology " + (t + 1) + " (Without Momentum)");
            for (int epoch = 0; epoch < trainLossNoMomentum.size(); epoch++) {
                noMomentumEpochSeries.add(epoch + 1, trainLossNoMomentum.get(epoch));
            }
            noMomentumEpochLossDataset.addSeries(noMomentumEpochSeries);

            if (testMseNoMomentum < bestNoMomentumMse) {
                bestNoMomentumMse = testMseNoMomentum;
                bestNoMomentumTopology = topology;
            }
        }

        // Plot results
        plotFinalLossGraph(momentumFinalLossSeries, noMomentumFinalLossSeries);
        plotEpochLossGraphs(momentumEpochLossDataset, "Epoch-by-Epoch Loss (With Momentum)");
        plotEpochLossGraphs(noMomentumEpochLossDataset, "Epoch-by-Epoch Loss (Without Momentum)");
    }
    
    private static void plotFinalLossGraph(XYSeries momentumSeries, XYSeries noMomentumSeries) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(momentumSeries);
        dataset.addSeries(noMomentumSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Final Loss Comparison: Momentum vs No-Momentum",
            "Topology Index",
            "Final Loss (MSE)",
            dataset,
            org.jfree.chart.plot.PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        JFrame frame = new JFrame("Final Loss Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    private static void plotEpochLossGraphs(XYSeriesCollection epochLossDataset, String title) {
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Epoch",
            "Loss (MSE)",
            epochLossDataset,
            org.jfree.chart.plot.PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }
    

	 // Train the network and return training losses
	 private static List<Double> trainNetworkWithLossTracking(NeuralNetwork<?> network, DataSet trainData, int epochs, double learningRate, double momentum) {
	     MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
	     mbp.setLearningRate(learningRate);
	     mbp.setMomentum(momentum);
	     mbp.setMaxIterations(epochs);
	
	     // Perform one full initialization step to ensure weightTrainingData is initialized
	     mbp.setMaxIterations(1);
	     network.learn(trainData); // This initializes internal structures
	     
	     List<Double> losses = new ArrayList<>();
	
	     for (int epoch = 0; epoch < epochs; epoch++) {
	         mbp.doOneLearningIteration(trainData);
	         double mse = MseCalculator.testNetwork(network, trainData);
	         losses.add(mse);
	     }
	
	     return losses;
	 }

    private static void trainAndTest(DataSet trainData, DataSet testData, boolean withMomentum) {
        int[] topology = withMomentum ? bestMomentumTopology : bestNoMomentumTopology;
        NeuralNetwork<?> network = createNetwork(topology, 3, 1);
        double momentum = withMomentum ? 0.9 : 0.0;
        trainNetwork(network, trainData, 100, 0.1, momentum);

        double testMse = MseCalculator.testNetwork(network, testData);
        System.out.println("\nTraining completed.");
        System.out.println("Using Topology: " + topologyToString(topology));
        System.out.println("Test MSE: " + testMse);
    }

    private static void epochByEpochEvaluation(DataSet trainData, DataSet testData) {
        // Keep epoch-by-epoch evaluation using the best no momentum topology
        int[] topology = bestNoMomentumTopology;
        NeuralNetwork<?> network = createNetwork(topology, 3, 1);

        MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
        mbp.setLearningRate(0.1); // Increased learning rate for better results
        mbp.setMomentum(0.0);     // Always with momentum for this evaluation

        mbp.learn(trainData); // Initialize weightTrainingData

        System.out.println("\nEpoch-by-Epoch Evaluation:");
        for (int epoch = 1; epoch <= 50; epoch++) {
            mbp.doOneLearningIteration(trainData);
            double trainMse = MseCalculator.testNetwork(network, trainData);
            double testMse = MseCalculator.testNetwork(network, testData);
            System.out.printf("Epoch %d: Train MSE = %.6f, Test MSE = %.6f%n", epoch, trainMse, testMse);
        }
    }

    private static void trainAndSingleTest(DataSet trainData, Scanner scanner) {
        // Keep single test using the best momentum topology
        int[] topology = bestMomentumTopology;
        NeuralNetwork<?> network = createNetwork(topology, 3, 1);

        // Train the network
        MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
        mbp.setLearningRate(0.1);
        mbp.setMomentum(0.9);
        mbp.setMaxIterations(100);
        network.learn(trainData);

        System.out.println("\nTraining completed.");

        // Ask user for input values
        System.out.print("Enter education level: ");
        double education = scanner.nextDouble();
        System.out.print("Enter years of experience: ");
        double experience = scanner.nextDouble();
        System.out.print("Enter gender (0 for female, 1 for male): ");
        double gender = scanner.nextDouble();

        // Prepare input and calculate output
        double[] input = {education, experience, gender};
        network.setInput(input);
        network.calculate();
        double[] output = network.getOutput();

        // Display the predicted result
        System.out.printf("Predicted salary: %.2f%n", output[0]);
    }

    private static void kFoldCrossValidation(DataSet dataSet, Scanner scanner) {
        System.out.print("\nEnter the number of folds (k): ");
        int k = scanner.nextInt();

        if (k < 2 || k > dataSet.size()) {
            System.out.println("Invalid value for k. Please enter a value between 2 and " + dataSet.size());
            return;
        }

        // Shuffle the dataset
        List<DataSetRow> rows = new ArrayList<>(dataSet.getRows());
        Collections.shuffle(rows);

        // Split the dataset into k folds
        int foldSize = dataSet.size() / k;
        List<DataSet> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            int start = i * foldSize;
            int end = (i == k - 1) ? rows.size() : start + foldSize; // Handle remainder
            DataSet fold = new DataSet(dataSet.getInputSize(), dataSet.getOutputSize());
            for (int j = start; j < end; j++) {
                fold.addRow(rows.get(j));
            }
            folds.add(fold);
        }

        double totalTrainMse = 0;
        double totalTestMse = 0;

        System.out.println("\nPerforming " + k + "-Fold Cross-Validation...");

        // Use momentum for K-fold as previously done
        for (int i = 0; i < k; i++) {
            // Prepare training and testing datasets
            DataSet testFold = folds.get(i);
            DataSet trainFolds = new DataSet(dataSet.getInputSize(), dataSet.getOutputSize());

            for (int j = 0; j < k; j++) {
                if (j != i) {
                    for (DataSetRow row : folds.get(j).getRows()) {
                        trainFolds.addRow(row);
                    }
                }
            }

            // Use same fixed topology {10, 5} for K-Fold
            int[] topology = bestMomentumTopology;
            NeuralNetwork<?> network = createNetwork(topology, 3, 1);
            trainNetwork(network, trainFolds, 100, 0.01, 0.9); // With momentum

            double trainMse = MseCalculator.testNetwork(network, trainFolds);
            double testMse = MseCalculator.testNetwork(network, testFold);

            totalTrainMse += trainMse;
            totalTestMse += testMse;

            System.out.printf("Fold %d: Train MSE = %.6f, Test MSE = %.6f%n", i + 1, trainMse, testMse);
        }

        System.out.printf("\nAverage Train MSE: %.6f%n", totalTrainMse / k);
        System.out.printf("Average Test MSE: %.6f%n", totalTestMse / k);
    }

    private static NeuralNetwork<?> createNetwork(int[] hiddenLayers, int inputSize, int outputSize) {
        int totalLayers = hiddenLayers.length + 2; // Input + hidden(s) + output
        int[] layers = new int[totalLayers];
        layers[0] = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            layers[i + 1] = hiddenLayers[i];
        }
        layers[totalLayers - 1] = outputSize;
        return new MultiLayerPerceptron(TransferFunctionType.SIGMOID, layers);
    }

    private static void trainNetwork(NeuralNetwork<?> network, DataSet trainData, int epochs, double learningRate, double momentum) {
        MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
        mbp.setLearningRate(learningRate);
        mbp.setMomentum(momentum);
        mbp.setMaxIterations(epochs);
        network.learn(trainData);
    }

    private static String topologyToString(int[] topology) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < topology.length; i++) {
            sb.append(topology[i]);
            if (i < topology.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}