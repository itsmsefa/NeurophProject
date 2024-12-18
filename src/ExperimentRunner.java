import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

public class ExperimentRunner {
    public static NeuralNetwork<?> runExperimentsAndFindBest(DataSet trainData, DataSet testData) {
        int[][] topologies = {
            {5},
            {10},
            {10,10},
            {20},
            {20,10},
            {10,5},
            {15,15},
            {5,5,5},
            {10,10,10},
            {20,20}
        };

        double bestMse = Double.MAX_VALUE;
        NeuralNetwork<?> bestNet = null;

        for (int[] topo : topologies) {
            NeuralNetwork<?> net = createMLP(topo, trainData.getInputSize(), trainData.getOutputSize());
            TrainHelper.trainNetwork(net, trainData, 100, 0.01, 0.9);
            double testMse = MseCalculator.testNetwork(net, testData);
            System.out.println("Topology " + arrToString(topo) + " Test MSE: " + testMse);
            if (testMse < bestMse) {
                bestMse = testMse;
                bestNet = createMLP(topo, trainData.getInputSize(), trainData.getOutputSize());
            }
        }

        System.out.println("Best topology found, Test MSE: " + bestMse);
        return bestNet;
    }

    public static NeuralNetwork<?> createMLP(int[] hiddenLayers, int inputSize, int outputSize) {
        int totalLayers = hiddenLayers.length + 2;
        int[] layers = new int[totalLayers];
        layers[0] = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            layers[i+1] = hiddenLayers[i];
        }
        layers[totalLayers-1] = outputSize;

        return new MultiLayerPerceptron(TransferFunctionType.SIGMOID, layers);
    }

    public static NeuralNetwork<?> createNetworkCopy(NeuralNetwork<?> net) {
        if (!(net instanceof MultiLayerPerceptron)) {
            throw new RuntimeException("Expected MultiLayerPerceptron");
        }
        MultiLayerPerceptron mlp = (MultiLayerPerceptron) net;
        int[] layerSizes = new int[mlp.getLayers().size()];
        for (int i = 0; i < mlp.getLayers().size(); i++) {
            layerSizes[i] = mlp.getLayers().get(i).getNeuronsCount();
        }
        return new MultiLayerPerceptron(TransferFunctionType.SIGMOID, layerSizes);
    }

    private static String arrToString(int[] arr) {
        if (arr.length == 0) return "[NoHidden]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i=0; i<arr.length; i++){
            sb.append(arr[i]);
            if(i!=arr.length-1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }
}