package ucr.cs.dblab.textclassifier;

import java.sql.SQLException;

import weka.core.Instance;
import weka.core.Instances;

public abstract class TextDatasetBalancer
{
	protected int mostCommon;
	protected int[] classCounts;
	
	public abstract Instances balance(Instances data) throws SQLException;
	
	protected void countClasses(Instances data)
	{
		int classValue;
		
		mostCommon = -1;
		classCounts = new int[data.numClasses()];
		
		for(Instance instance : data)
		{
			classValue = (int) instance.classValue();
			classCounts[classValue]++;
			mostCommon = mostCommon < 0 || classCounts[classValue] > classCounts[mostCommon] ? classCounts[classValue] : mostCommon;
		}
	}
}
