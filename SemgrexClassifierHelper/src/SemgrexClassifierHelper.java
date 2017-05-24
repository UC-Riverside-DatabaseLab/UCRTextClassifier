import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class SemgrexClassifierHelper
{
	private final MaxentTagger tagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
	private final DependencyParser parser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
	private final JSONParser jsonParser = new JSONParser();
	private Map<List<HasWord>, SemanticGraph> semanticGraphs = new HashMap<List<HasWord>, SemanticGraph>();
	private Map<String, Double> distribution = new HashMap<String, Double>();
	private Map<SemgrexPattern, String> semgrexPatterns;
	
	public SemanticGraph buildSemanticGraph(List<HasWord> sentence)
	{
		if(semanticGraphs.containsKey(sentence))
		{
			return semanticGraphs.get(sentence);
		}
		
		SemanticGraph semanticGraph = new SemanticGraph(parser.predict(tagger.tagSentence(sentence)).typedDependencies());
		
		semanticGraphs.put(sentence, semanticGraph);
		return semanticGraph;
	}
	
	private void classifyText(String text)
	{
		SemanticGraph semanticGraph;
		
		distribution.clear();
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(Entry<SemgrexPattern, String> entry : semgrexPatterns.entrySet())
			{
				if(entry.getKey().matcher(semanticGraph).find())
				{
					if(!distribution.containsKey(entry.getValue()))
					{
						distribution.put(entry.getValue(), 0.0);
					}
					
					distribution.put(entry.getValue(), distribution.get(entry.getValue()) + 1.0);
				}
			}
		}
		
		System.out.println(((JSONObject) distribution).toJSONString());
	}
	
	private void parseText(String text)
	{
		ArrayList<String> jsonArray = new ArrayList<String>();
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			jsonArray.add(buildSemanticGraph(sentence).toFormattedString());
		}
		
		System.out.println(((JSONArray) jsonArray).toJSONString());
	}
	
	public void receiveCommand(String json) throws ParseException
	{
		JSONObject jsonObject = (JSONObject) jsonParser.parse(json);
		
		if(jsonObject.containsKey("mode"))
		{
			String mode = (String) jsonObject.get("mode");
			
			if(mode.equals("init"))
			{
				semgrexPatterns = new HashMap<SemgrexPattern, String>();
			}
			else if(mode.equals("parse"))
			{
				parseText((String) jsonObject.get("text"));
			}
			else if(mode.equals("add_pattern"))
			{
				semgrexPatterns.put(SemgrexPattern.compile((String) jsonObject.get("pattern")), (String) jsonObject.get("class"));
			}
			else if(mode.equals("classify"))
			{
				classifyText((String) jsonObject.get("text"));
			}
		}
	}
	
	public static void main(String[] args)
	{
		SemgrexClassifierHelper semgrexClassifierHelper = new SemgrexClassifierHelper();
		Scanner scanner = new Scanner(System.in);
		
		try
		{
			while(true)
			{
				semgrexClassifierHelper.receiveCommand(scanner.nextLine());
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		
		scanner.close();
	}
}
