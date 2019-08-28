/**
 * 
 */
package com.neuronalstructuressegmentation;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.toIntExact;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class Segmentation {

	protected static final Logger log = LoggerFactory.getLogger(Segmentation.class);
	
	Params params = new Params();

  	private long startTrainTime = 0;
  	private long endTrainTime = 0;
  	protected static boolean save = true;
	
	/**
	 * @param args
	 */
	
	public void execute(String[] args) throws Exception {
		
		// Loading the data		
		System.out.println("Loading data....");		
		CustomPathLabelGenerator labelMaker = new CustomPathLabelGenerator();
		
		File mainTrainDataPath = new File(System.getProperty("user.dir"), "/src/main/resources/SplitData/Train/");
		File mainTestDataPath = new File(System.getProperty("user.dir"), "/src/main/resources/SplitData/Test/");
		
		FileSplit trainFileSplit = new FileSplit(mainTrainDataPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		FileSplit testFileSplit = new FileSplit(mainTestDataPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		
		int numExamples = toIntExact(trainFileSplit.length());
		int numTest = toIntExact(testFileSplit.length());		
		log.info("Number of images: " + numExamples);
		log.info("Number of Test: " + numTest);
		
		RandomPathFilter trainPathFilter = new RandomPathFilter(params.getRng(), null, numExamples);
		RandomPathFilter testPathFilter = new RandomPathFilter(params.getRng(), null, numTest);
		
		InputSplit[] trainInputSplit = trainFileSplit.sample(trainPathFilter);
		InputSplit[] testInputSplit = testFileSplit.sample(testPathFilter);
		
		InputSplit trainData = trainInputSplit[0];
		InputSplit testData = testInputSplit[0];		
		log.info("Train data: " + trainData.length());
		log.info("Test data: " + testData.length());
		
		// Data augmentation
		// What do we need? Shift, Rotation, Deformation
                ImageTransform warpTransform = new WarpImageTransform(42);      
                ImageTransform rotateTransform1 = new RotateImageTransform(30);
                ImageTransform rotateTransform2 = new RotateImageTransform(60);
                ImageTransform rotateTransform3 = new RotateImageTransform(60);
                ImageTransform scaleTransform = new ScaleImageTransform(5);
                boolean shuffle = true;
                List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(warpTransform,0.9),
        		                                                   new Pair<>(rotateTransform1,0.9),
        		                                                   new Pair<>(rotateTransform2,0.9),
        		                                                   new Pair<>(rotateTransform3,0.9),
        		                                                   new Pair<>(scaleTransform,0.9));        
                ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);       
        
		ImageRecordReader trainRecordReader = new ImageRecordReader(params.getHeight(), 
				                                                    params.getWidth(), 
				                                                    params.getChannels(), 
				                                                    labelMaker); 
		ImageRecordReader testRecordReader = new ImageRecordReader(params.getHeight(), 
				                                                   params.getWidth(), 
				                                                   params.getChannels(),
				                                                   labelMaker); 
		
		trainRecordReader.initialize(trainData, transform); 
		testRecordReader.initialize(testData);
			
		DataNormalization scaler = new ExtendedImagePreProcessingScaler(0, 1); 
		
		DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, 
				                                                        params.getBatch(), 
				                                                        params.getLabelIndex(), 
				                                                        params.getLabelIndex(), 
				                                                        true); // true arg just means "don't try to convert label to a one-hot array" like we need for classification			
		scaler.fit(trainDataIter);
                trainDataIter.setPreProcessor(scaler); 
               
                DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, 
        		                                                       params.getBatch(), 
        		                                                       params.getLabelIndex(), 
        		                                                       params.getLabelIndex(), 
        		                                                       true);
                scaler.fit(testDataIter);       
                testDataIter.setPreProcessor(scaler);
		
		// Building model...        
                log.info("Building model....");        
                ComputationGraph network;
		String modelFilename = "segmentationU-netModel.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading existing model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
		} else {
			network = new NetworkConfig().getNetworkConfig();
			network.addListeners(new ScoreIterationListener(1));
			network.init();
			log.info(network.summary(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels())));
        	
		        //Configuring early stopping
			EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
					.epochTerminationConditions(new MaxEpochsTerminationCondition(params.getEpochs()))
					.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(params.getMaxTimeIterTerminationCondition(), TimeUnit.HOURS))
					.scoreCalculator(new DataSetLossCalculator(testDataIter, true))
					.evaluateEveryNEpochs(1)
					.modelSaver(new LocalFileGraphSaver(new File(System.getProperty("user.dir")).toString()))
					.build();
			
			EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, network, trainDataIter);
			
			// Conducting early stopping training of the model...
			log.info("Training model....");
			EarlyStoppingResult<ComputationGraph> result = trainer.fit();
	        
	                // Saving the best model...	        
	                log.info("Saving the best model....");
	                //Get the best model:
	                network = result.getBestModel();
	                if (save) {
	        	       ModelSerializer.writeModel(network, modelFilename, true);
	                }
	                System.out.println();
	                log.info("The best model has been saved....");
	                System.out.println();
	        
	                //Getting the results...
			System.out.println("Termination reason: " + result.getTerminationReason());
			System.out.println("Termination details: " + result.getTerminationDetails());
			System.out.println("Total epochs: " + result.getTotalEpochs());
			System.out.println("Best epoch number: " + result.getBestModelEpoch());
			System.out.println("Score at best epoch: " + result.getBestModelScore());
		}
        
	}
		
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		new Segmentation().execute(args);
	}
}
