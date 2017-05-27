import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
		
		//semanticGraphs.put(sentence, semanticGraph);
		return semanticGraph;
	}
	
	private String classifyText(String text)
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
		
		return ((JSONObject) distribution).toJSONString();
	}
	
	private String parseText(String text)
	{
		List<String> sentences = new ArrayList<String>();
		String formatted;
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			formatted = buildSemanticGraph(sentence).toFormattedString().replace("\n", " ");
			
			while(formatted.contains("  "))
			{
				formatted = formatted.replace("  ", " ");
			}
			
			sentences.add(formatted);
		}
		
		return JSONArray.toJSONString(sentences);
	}
	
	public String receiveCommand(String json) throws IOException, ParseException
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
				return parseText((String) jsonObject.get("text"));
			}
			else if(mode.equals("add_pattern"))
			{
				semgrexPatterns.put(SemgrexPattern.compile((String) jsonObject.get("pattern")), (String) jsonObject.get("class"));
			}
			else if(mode.equals("classify"))
			{
				return classifyText((String) jsonObject.get("text"));
			}
		}
		
		return null;
	}
	
	public static void main(String[] args) throws IOException, ParseException
	{
		SemgrexClassifierHelper semgrexClassifierHelper = new SemgrexClassifierHelper();
		ServerSocket serverSocket = new ServerSocket(9000);
		Socket clientSocket = serverSocket.accept();
	    PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
	    BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
	    String inputLine, outputLine;
	    
	    while((inputLine = in.readLine()) != null)
	    {
	    	outputLine = semgrexClassifierHelper.receiveCommand(inputLine);
	    	
	    	if(outputLine != null)
	    	{
	    		out.println(outputLine.replace("\n", "__NEWLINE__"));
	    	}
	    }
	    
	    serverSocket.close();
	}
}
