package ucr.cs.dblab.textclassifier;

import weka.core.Instances;

public class GoogleDatasetBalancer extends TextDatasetBalancer
{
	private String[] proxies;
	private String engineID;
	
	public GoogleDatasetBalancer(String[] proxies, String engineID)
	{
		this.proxies = proxies;
		this.engineID = engineID;
	}
	
	public Instances balance(Instances data)
	{
		return null;
	}
}