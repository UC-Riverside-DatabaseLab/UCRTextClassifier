package ucr.cs.dblab.textclassifier;

import java.sql.SQLException;
import java.util.Map;

import weka.core.Instances;

public class EnsembleTextDatasetBalancer extends TextDatasetBalancer
{
	public static enum Parameter{PPDB, GOOGLE, DUPLICATE};
	private TextDatasetBalancer ppdbDatasetBalancer = null, googleDatasetBalancer = null, duplicateDatasetBalancer = null;
	
	public EnsembleTextDatasetBalancer(Map<Parameter, Map<Object, Object>> parameters)
	{
		for(Parameter parameter : Parameter.values())
		{
			switch(parameter)
			{
				case PPDB:
					ppdbDatasetBalancer = new PPDBDatasetBalancer((Map<Object, Object>) parameters.get(Parameter.PPDB));
					break;
				case GOOGLE:
					break;
				case DUPLICATE:
					duplicateDatasetBalancer = new DuplicateDatasetBalancer();
					break;
				default:
					break;
			}
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
