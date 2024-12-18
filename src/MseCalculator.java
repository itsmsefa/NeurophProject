public class MseCalculator {
    public static double testNetwork(org.neuroph.core.NeuralNetwork<?> network, org.neuroph.core.data.DataSet testSet) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < testSet.size(); i++) {
            double[] input = testSet.getRowAt(i).getInput();
            double[] desired = testSet.getRowAt(i).getDesiredOutput();

            network.setInput(input);
            network.calculate();
            double[] output = network.getOutput();

            for (int j = 0; j < desired.length; j++) {
                double diff = desired[j] - output[j];
                sum += diff * diff;
                count++;
            }
        }
        return sum / count;
    }
}