package ucr.cs.dblab.textclassifier;

import java.sql.SQLException;
import java.util.Map;

import ucr.cs.dblab.textclassifier.PPDBDatasetBalancer.Parameter;
import weka.core.Instances;

public class EnsembleTextDatasetBalancer extends TextDatasetBalancer
{
	private TextDatasetBalancer ppdbDatasetBalancer = null, googleDatasetBalancer = null, duplicateDatasetBalancer = null;
	
	public EnsembleTextDatasetBalancer(Map<Parameter, Object> parameters)
	{
		boolean ppdb = true;
		
		for(Parameter parameter : PPDBDatasetBalancer.Parameter.values())
		{
			if(!parameters.containsKey(parameter))
			{
				ppdb = false;
				break;
			}
		}
		
		if(ppdb)
		{
			ppdbDatasetBalancer = new PPDBDatasetBalancer(parameters);
		}
	}
	
	public Instances balance(Instances data) throws SQLException
	{
		Instances newData = new Instances(data, 0);
		
		if(ppdbDatasetBalancer != null)
		{
			newData.addAll(ppdbDatasetBalancer.balance(newData));
		}
		
		if(googleDatasetBalancer != null)
		{
			newData.addAll(googleDatasetBalancer.balance(newData));
		}

		if(duplicateDatasetBalancer != null)
		{
			newData.addAll(duplicateDatasetBalancer.balance(newData));
		}
		
		return newData;
	}
	
	public static void main(String[] args)
	{
		
	}
}
