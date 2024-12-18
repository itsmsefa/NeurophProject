import java.util.ArrayList;
import java.util.Collections;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public class KFolds {
    public static void runKFolds(DataSet combinedData, int k, NeuralNetwork<?> baseNetwork, boolean useMomentum) {
        int size = combinedData.size();
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < size; i++) indices.add(i);
        Collections.shuffle(indices);

        int foldSize = size / k;
        double totalTrainMse = 0.0;
        double totalTestMse = 0.0;

        for (int i = 0; i < k; i++) {
            int testStart = i * foldSize;
            int testEnd = (i == k-1) ? size : testStart + foldSize;
            DataSet testSet = extractSubset(combinedData, indices, testStart, testEnd);
            DataSet trainSet = combineSubsets(combinedData, indices, 0, testStart, testEnd, size);

            NeuralNetwork<?> netCopy = ExperimentRunner.createNetworkCopy(baseNetwork);
            // Just use normal training for K-Fold (no epoch by epoch needed)
            TrainHelper.trainNetwork(netCopy, trainSet, 100, 0.01, useMomentum ? 0.9 : 0.0);

            double trainMse = MseCalculator.testNetwork(netCopy, trainSet);
            double testMse = MseCalculator.testNetwork(netCopy, testSet);

            totalTrainMse += trainMse;
            totalTestMse += testMse;
        }

        System.out.println("Average Train MSE: " + (totalTrainMse / k));
        System.out.println("Average Test MSE: " + (totalTestMse / k));
    }

    private static DataSet extractSubset(DataSet data, ArrayList<Integer> indices, int start, int end) {
        DataSet subset = new DataSet(data.getInputSize(), data.getOutputSize());
        for (int i = start; i < end && i < indices.size(); i++) {
            DataSetRow existingRow = data.getRowAt(indices.get(i));
            subset.addRow(new DataSetRow(existingRow.getInput(), existingRow.getDesiredOutput()));
        }
        return subset;
    }

    private static DataSet combineSubsets(DataSet data, ArrayList<Integer> indices, int start1, int end1, int start2, int end2) {
        DataSet subset = new DataSet(data.getInputSize(), data.getOutputSize());
        for (int i = start1; i < end1 && i < indices.size(); i++) {
            DataSetRow existingRow = data.getRowAt(indices.get(i));
            subset.addRow(new DataSetRow(existingRow.getInput(), existingRow.getDesiredOutput()));
        }
        for (int i = start2; i < end2 && i < indices.size(); i++) {
            DataSetRow existingRow = data.getRowAt(indices.get(i));
            subset.addRow(new DataSetRow(existingRow.getInput(), existingRow.getDesiredOutput()));
        }
        return subset;
    }
}