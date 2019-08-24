/**
 * 
 */
package com.neuronalstructuressegmentation;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.toIntExact;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
		CustomPathLabelGenerator multiPathlabelMaker = new CustomPathLabelGenerator();
		
		File mainTrainDataPath = new File(System.getProperty("user.dir"), "/src/main/resources/SplitData/Train/");
		File mainTestDataPath = new File(System.getProperty("user.dir"), "/src/main/resources/SplitData/Test/");
		
		FileSplit trainFileSplit = new FileSplit(mainTrainDataPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		FileSplit testFileSplit = new FileSplit(mainTestDataPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		
		int numExamples = toIntExact(trainFileSplit.length());
		int numTest = toIntExact(testFileSplit.length());		
		
		RandomPathFilter trainPathFilter = new RandomPathFilter(params.getRng(), null, numExamples);
		RandomPathFilter testPathFilter = new RandomPathFilter(params.getRng(), null, numTest);
		
		InputSplit[] trainInputSplit = trainFileSplit.sample(trainPathFilter);
		InputSplit[] testInputSplit = testFileSplit.sample(testPathFilter);
		
		InputSplit trainData = trainInputSplit[0];
		InputSplit testData = testInputSplit[0];		
		
		// Data transformation/augmentation
		// But data augmentation isn't enough yet. Still need shift/translation...
        ImageTransform warpTransform = new WarpImageTransform(params.getRng(), 42);      
        ImageTransform rotateTransform = new RotateImageTransform(params.getRng(), 42);
        boolean shuffle = true;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(warpTransform,0.9),
        		                                                   new Pair<>(rotateTransform,0.9));        
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);       
        
		ImageRecordReader trainRecordReader = new ImageRecordReader(params.getHeight(), 
				                                                    params.getWidth(), 
				                                                    params.getChannels(), 
				                                                    multiPathlabelMaker); 
		ImageRecordReader testRecordReader = new ImageRecordReader(params.getHeight(), 
				                                                   params.getWidth(), 
				                                                   params.getChannels()); 
		
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
		String modelFilename = "segmentationModel.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading existing model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
		} else {
			network = new NetworkConfig().getNetworkConfig();
			network.addListeners(new PerformanceListener(1), new ScoreIterationListener(1));;
			network.init();
			log.info(network.summary(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels())));
        	
        	// Training model...       	
            log.info("Training model....");           
            startTrainTime = System.currentTimeMillis();
            for (int i = 1; i <= params.getEpochs(); i++) {
            	trainDataIter.reset();
            	while (trainDataIter.hasNext()) {
            		network.fit(trainDataIter);
            	}
            	log.info("*** Epoch " + i + " completed." );
            }
            endTrainTime = System.currentTimeMillis();
            log.info("****************End of Training********************");
	        System.out.println();
	        log.info("Training time : " + (endTrainTime - startTrainTime) / 60000.0 + " min");
	        System.out.println();
	        
	        // Saving saving model...	        
	        log.info("Saving model....");	        
	        if (save) {
	        	ModelSerializer.writeModel(network, modelFilename, true);
	        }
	        System.out.println();
	        log.info("Model has been saved....");
		}
		
		// Evaluating model...       
        //log.info("Evaluating model....");
        
	}
		
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		new Segmentation().execute(args);
	}
}
