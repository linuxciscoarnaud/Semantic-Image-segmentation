/**
 * 
 */
package com.neuronalstructuressegmentation;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * @author Arnaud
 *
 */

public class Params {

	// Parameters for network configuration
	private OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
	private Activation activation = Activation.RELU;
	private WeightInit weightInit = WeightInit.RELU;
	private IUpdater updater = new AdaDelta();
	private CacheMode cacheMode = CacheMode.NONE;
	private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
	private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
	
	// Parameters for the training phase
	private int epochs = 100;
	protected static int maxTimeIterTerminationCondition = 48; // 48 hours
	private int batch = 1; // batch is reduced to a single image (Training section of the paper)
	
	// Parameters for input/output data
	private int height = 512;
	private int width = 512;
	private int channels = 1;
	private int labelIndex = 1; // We have 2 Writables (columns); index 0 is features image NDArrayWritable, index 1 is labels image NDArrayWritable
	
	private long seed = 123; // Integer for reproducibility of a random number generator
	private Random rng = new Random(seed);
	
	// Getters
	
	public OptimizationAlgorithm getOptimizationAlgorithm() {
		return optimizationAlgorithm;
	}
	
	public Activation getActivation() {
		return activation;
	}
	
	public WeightInit getWeightInit() {
		return weightInit;
	}
	
	public IUpdater getUpdater() {
		return updater;
	}
	
	public CacheMode getCacheMode() {
		return cacheMode;
	}
	
	public WorkspaceMode getWorkspaceMode() {
		return workspaceMode;
	}
	
	public ConvolutionLayer.AlgoMode getCudnnAlgoMode() {
		return cudnnAlgoMode;
	}
	
	public int getEpochs() {
		return epochs;
	}
	
	public int getMaxTimeIterTerminationCondition() {
 		return maxTimeIterTerminationCondition;
 	}
	
	public int getBatch() {
		return batch ;
	}
	
	public int getHeight() {
		return height;
	}
	
	public int getWidth( ) {
		return width;
	}
	
	public int getChannels() {
		return channels;
	}
	
	public int getLabelIndex() {
		return labelIndex;
	}
	
	public long getSeed() {
		return seed;
	}
	
	public Random getRng() {
		return rng;
	}
}
