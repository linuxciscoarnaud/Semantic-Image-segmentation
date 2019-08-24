/**
 * 
 */
package com.neuronalstructuressegmentation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * @author Arnaud
 *
 */

public class ExtendedImagePreProcessingScaler extends ImagePreProcessingScaler {

	public ExtendedImagePreProcessingScaler(double minRange, double maxRange) {
		super(minRange, maxRange);
	}
	
	@Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();
        INDArray labels = toPreProcess.getLabels(); // added
        this.preProcess(features);
        this.preProcess(labels); // added
    }
}
