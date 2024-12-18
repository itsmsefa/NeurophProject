import net.sourceforge.jFuzzyLogic.FIS;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;



class DataGeneration {
	public static void main(String[] args) {

        // FCL dosyasının yolu
        String filename = "src/salary.fcl";

        // FIS objesini yükle
        FIS fis = FIS.load(filename, true);
        if (fis == null) {
            System.err.println("FCL dosyası yüklenemedi: " + filename);
            return;
        }

        // Rastgele sayı üretici
        Random rand = new Random();

        // CSV dosyasına yazma
        String csvFile = "src/data.csv";
        try (FileWriter writer = new FileWriter(csvFile)) {
            // Başlık satırı
            writer.append("education,experience,gender,salary\n");

            // 4000 satır üret
            for (int i = 0; i < 4000; i++) {
                // Rastgele education [0..20]
                int education = rand.nextInt(21); // 0 dahil 20 dahil
                // Rastgele experience [0..25]
                int experience = rand.nextInt(26); // 0 dahil 25 dahil
                // Rastgele gender [0..1] (0 female, 1 male)
                int gender = rand.nextInt(2); // 0 veya 1

                // FIS girişlerini set et
                fis.setVariable("education", education);
                fis.setVariable("experience", experience);
                fis.setVariable("gender", gender);

                // Çıktıyı hesapla
                fis.evaluate();

                // Maaş değerini al
                double salary = fis.getVariable("salary").getValue();

                // CSV satırı yaz
                writer.append(String.valueOf(education)).append(",")
                      .append(String.valueOf(experience)).append(",")
                      .append(String.valueOf(gender)).append(",")
                      .append(String.valueOf(salary)).append("\n");
            }

            System.out.println("Veriler başarılı bir şekilde " + csvFile + " dosyasına yazıldı.");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
