package ucr.cs.dblab.textclassifier;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PPDBDatasetBalancer extends TextDatasetBalancer
{
	public static enum Parameter{HOST, PORT, DATABASE, USER, PASSWORD, SCORING_FEATURE, THRESHOLD, MIN_RANGE, MAX_RANGE};
	private String host, database, user, password, scoringFeature;
	private int port, minRange, maxRange;
	private double threshold;
	
	public PPDBDatasetBalancer(Map<Object, Object> parameters)
	{
		this.host = (String) parameters.get(Parameter.HOST);
		this.port = (Integer) parameters.get(Parameter.PORT);
		this.database = (String) parameters.get(Parameter.DATABASE);
		this.user = (String) parameters.get(Parameter.USER);
		this.password = (String) parameters.get(Parameter.PASSWORD);
		this.scoringFeature = parameters.containsKey(Parameter.SCORING_FEATURE) ? (String) parameters.get(Parameter.SCORING_FEATURE) : "PPDB2SCore";
		this.threshold = parameters.containsKey(Parameter.THRESHOLD) ? (Double) parameters.get(Parameter.THRESHOLD) : 0.0;
		this.minRange = parameters.containsKey(Parameter.MIN_RANGE) ? Math.max(1, (Integer) parameters.get(Parameter.MIN_RANGE)) : 1;
		this.maxRange = parameters.containsKey(Parameter.MAX_RANGE) ? Math.max(1, (Integer) parameters.get(Parameter.MAX_RANGE)) : 1;
		
		if(minRange > maxRange)
		{
			int temp = minRange;
			
			minRange = maxRange;
			maxRange = temp;
		}
	}
	
	public Instances balance(Instances data) throws SQLException
	{
		countClasses(data);
		
		Map<String, List<String>> paraphrases = new HashMap<String, List<String>>();
		Instances newInstances = new Instances(data, 0);
		String[] words;
		String phrase, text;
		Instance newInstance;
		Connection connection = DriverManager.getConnection("jdbc:mysql://" + host + ":" + port + "/" + database, user, password);
		int classValue, minCutoff, maxCutoff, length, index;
		
		for(Instance instance : data)
		{
			classValue = (int) instance.classValue();
			
			if(classCounts[classValue] == classCounts[mostCommon])
			{
				continue;
			}
			
			words = instance.stringValue(0).split(" ");
			
			paraphrases.clear();
			
			for(int i = 0; i < words.length; i++)
			{
				minCutoff = Math.min(words.length, i + minRange);
				maxCutoff = Math.min(words.length, i + maxRange);
				phrase = "";
				length = 0;
				
				for(int j = i; j < minCutoff; j++)
				{
					phrase += (phrase.isEmpty() ? "" : " ") + words[j];
					length++;
				}
				
				if(length < minRange)
				{
					break;
				}
				
				findParaphrases(phrase, paraphrases, connection);
				
				for(int j = i + length; j < maxCutoff; j++)
				{
					phrase += (phrase.isEmpty() ? "" : " ") + words[j];
					
					findParaphrases(phrase, paraphrases, connection);
				}
			}
			
			for(String ph : paraphrases.keySet())
			{
				index = instance.stringValue(0).indexOf(ph);
				text = instance.stringValue(0).replace(ph, "");
				
				for(String paraphrase : paraphrases.get(ph))
				{
					newInstance = new DenseInstance(instance);
					
					newInstance.setDataset(newInstances);
					newInstance.setValue(0, text.substring(0, index) + paraphrase + text.substring(index));
					newInstance.setValue(data.classIndex(), instance.classValue());
					newInstances.add(instance);
				}
			}
		}
		
		connection.close();
		return newInstances;
	}
	
	private void findParaphrases(String phrase, Map<String, List<String>> paraphrases, Connection connection) throws SQLException
	{
		if(paraphrases.containsKey(phrase))
		{
			return;
		}
		
		Statement statement = connection.createStatement();
		ResultSet resultSet = statement.executeQuery("SELECT paraphrase FROM ppdb WHERE phrase = '"
			+ phrase.replace("\\", "\\\\").replace("'", "\\'") + "' AND " + scoringFeature + " > " + threshold);
		
		while(resultSet.next())
		{
			if(!paraphrases.containsKey(phrase))
			{
				paraphrases.put(phrase, new ArrayList<String>());
			}
			
			paraphrases.get(phrase).add(resultSet.getString("paraphrase"));
		}
		
		resultSet.close();
		statement.close();
	}
}
