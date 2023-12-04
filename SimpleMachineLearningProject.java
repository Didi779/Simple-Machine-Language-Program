
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
package simplemachinelearningproject;

public class SimpleWekaExample {
    public static void main(String[] args) {
        try {
            // Load dataset (replace "path_to_dataset" with your dataset path)
            DataSource source = new DataSource("https://registry.opendata.aws/");
            Instances data = source.getDataSet();
            
            // Set class index (assuming the class is in the last attribute)
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            
            // Initialize classifier (using Logistic regression as an example)
            Classifier classifier = new Logistic();
            
            // Build model
            classifier.buildClassifier(data);
            
            // Evaluate model using cross-validation (10-fold cross-validation)
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new java.util.Random(1));
            
            // Output evaluation results
            System.out.println("=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
