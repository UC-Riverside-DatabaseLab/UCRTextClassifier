package ucr.cs.dblab.textclassifier;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class DuplicateDatasetBalancer extends TextDatasetBalancer
{
	public Instances balance(Instances data)
	{
		countClasses(data);

		Instances newInstances = new Instances(data, classCounts[mostCommon] * data.numClasses());
		Instance newInstance;
		int classValue, numCopies;
		
		for(Instance instance : data)
		{
			classValue = (int) instance.classValue();
			
			if(classCounts[classValue] == classCounts[mostCommon])
			{
				continue;
			}
			
			numCopies = (int) Math.round((double) (classCounts[mostCommon] - classCounts[classValue]) / (double) classCounts[classValue]);
			
			for(int i = 0; i < numCopies; i++)
			{
				newInstance = new DenseInstance(instance);
				
				newInstance.setDataset(newInstances);
				newInstances.add(instance);
			}
		}
		
		return newInstances;
	}
}