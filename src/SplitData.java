import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

class SplitData {

	public static void main(String[] args) {

        String inputFile = "src/normalized_data.csv";
        String trainFile = "src/train.csv";
        String testFile = "src/test.csv";

        ArrayList<String> dataLines = new ArrayList<>();
        String header = "";

        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            // İlk satır başlık
            header = br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                dataLines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Listeyi karıştır
        Collections.shuffle(dataLines);

        // Toplam satır sayısı
        int total = dataLines.size();
        // %75
        int trainCount = (int) (0.75 * total);

        // Eğitim verisi (ilk %75)
        ArrayList<String> trainData = new ArrayList<>(dataLines.subList(0, trainCount));
        // Test verisi (geri kalan %25)
        ArrayList<String> testData = new ArrayList<>(dataLines.subList(trainCount, total));

        // Eğitim dosyasına yaz
        try (FileWriter fwTrain = new FileWriter(trainFile)) {
            fwTrain.append(header).append("\n");
            for (String d : trainData) {
                fwTrain.append(d).append("\n");
            }
            System.out.println("Eğitim verisi " + trainFile + " dosyasına yazıldı.");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Test dosyasına yaz
        try (FileWriter fwTest = new FileWriter(testFile)) {
            fwTest.append(header).append("\n");
            for (String d : testData) {
                fwTest.append(d).append("\n");
            }
            System.out.println("Test verisi " + testFile + " dosyasına yazıldı.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
