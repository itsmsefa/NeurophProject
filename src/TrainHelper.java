import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class TrainHelper {

    // Train for a fixed number of epochs
    public static void trainNetwork(NeuralNetwork<?> network, DataSet trainData, int epochs, double learningRate, double momentum) {
        // cast learning rule to MomentumBackpropagation which is a subclass of SupervisedLearning
        MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
        mbp.setLearningRate(learningRate);
        mbp.setMomentum(momentum);
        mbp.setMaxIterations(epochs);
        // This will train until maxIterations is reached
        network.learn(trainData);
    }

    // Train epoch by epoch manually and print MSE each epoch
    public static void trainNetworkEpochByEpoch(NeuralNetwork<?> network, DataSet trainData, DataSet testData, int epochs) {
        MomentumBackpropagation mbp = (MomentumBackpropagation) network.getLearningRule();
        mbp.setLearningRate(0.01);
        mbp.setMomentum(0.9);
        // We won't set maxIterations here because we manually control training

        // SupervisedLearning provides doOneLearningIteration(DataSet)
        // which trains the network for one iteration (epoch)
        for (int e = 1; e <= epochs; e++) {
            mbp.doOneLearningIteration(trainData); 
            double trainMse = MseCalculator.testNetwork(network, trainData);
            double testMse = MseCalculator.testNetwork(network, testData);
            System.out.println("Epoch " + e + ": Train MSE=" + trainMse + ", Test MSE=" + testMse);
        }
    }
}