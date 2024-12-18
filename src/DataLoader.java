import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class DataLoader {
    public static DataSet loadDataSet(String filename) {
        ArrayList<double[]> inputs = new ArrayList<>();
        ArrayList<double[]> outputs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // read header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                // Expecting exactly 4 columns: 3 inputs, 1 output
                double education = Double.parseDouble(parts[0]);
                double experience = Double.parseDouble(parts[1]);
                double gender = Double.parseDouble(parts[2]);
                double salary = Double.parseDouble(parts[3]);

                double[] inputArray = {education, experience, gender};
                double[] outputArray = {salary};

                inputs.add(inputArray);
                outputs.add(outputArray);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataSet ds = new DataSet(3, 1);
        for (int i = 0; i < inputs.size(); i++) {
            ds.addRow(new DataSetRow(inputs.get(i), outputs.get(i)));
        }
        return ds;
    }
}