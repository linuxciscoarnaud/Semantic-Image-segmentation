/**
 * 
 */
package com.neuronalstructuressegmentation;

import java.io.File;
import java.net.URI;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;

/**
 * @author Arnaud
 *
 */

public class CustomPathLabelGenerator implements PathLabelGenerator {

	NativeImageLoader imageLoader = new NativeImageLoader(512, 512, 1);
	Writable ndArrayWritable = null;
	
	public CustomPathLabelGenerator() {}
	
	public Writable getLabelForPath(String path) {		
		
		File labelFile = new File(System.getProperty("user.dir"), "/src/main/resources/SplitData/Labels/" + new File(path).getName());
		try {
			ndArrayWritable = new NDArrayWritable(imageLoader.asMatrix(labelFile));
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		return ndArrayWritable;
	}
	
	public Writable getLabelForPath(URI uri) {
		return getLabelForPath(new File(uri).toString());
	}
	
	public boolean inferLabelClasses() {
		return false;
	}
}
