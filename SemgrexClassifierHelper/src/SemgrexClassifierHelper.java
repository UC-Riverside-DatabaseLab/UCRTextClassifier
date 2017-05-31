import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.function.Predicate;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.Tree;

public class SemgrexClassifierHelper
{
	private static final String shutdownCommand = "__SHUTDOWN__";
	private static final String sbar = "SBAR";
	private static final Predicate<Tree> predicate = new Predicate<Tree>()
	{
		@Override
		public boolean test(Tree tree)
		{
			return !tree.value().equals(sbar);
		}
	};
	private final MaxentTagger tagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
	private final DependencyParser parser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
	private final LexicalizedParser constituencyParser = LexicalizedParser.loadModel();
	private final JSONParser jsonParser = new JSONParser();
	private Map<String, Double> distribution = new HashMap<String, Double>();
	private Map<SemgrexPattern, String> semgrexPatterns;
	private boolean splitSentences = false;
	
	public SemanticGraph buildSemanticGraph(List<HasWord> sentence)
	{
		return new SemanticGraph(parser.predict(tagger.tagSentence(sentence)).typedDependencies());
	}
	
	private String classifyText(String text)
	{
		SemanticGraph semanticGraph;
		
		distribution.clear();
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
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
		
		return JSONObject.toJSONString(distribution);
	}
	
	private List<Tree> parseClauses(Tree root)
	{
		Queue<Tree> queue = new LinkedList<Tree>();
		List<Tree> trees = new ArrayList<Tree>();
		String rootValue = root.value();
		Tree tree;
		
		root.setValue("ROOT");
		queue.add(root);
		
		while(!queue.isEmpty())
		{
			tree = queue.remove();
			
			for(Tree child : tree.children())
			{
				if(child.value().equals(sbar))
				{
					trees.addAll(parseClauses(child));
				}
				else
				{
					queue.add(child);
				}
			}
		}
		
		tree = root.prune(predicate);
		
		if(tree != null && tree.size() > 3)
		{
			trees.add(tree.deepCopy());
		}
		
		root.setValue(rootValue);
		return trees;
	}
	
	private List<List<HasWord>> parseSentences(String text)
	{
		List<List<HasWord>> sentences = new ArrayList<List<HasWord>>();
		StringBuilder stringBuilder = new StringBuilder(text.length());
		
		//System.out.println("Original text: " + text);
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			for(Tree tree : parseClauses(constituencyParser.apply(sentence)))
			{
				stringBuilder.delete(0, stringBuilder.length());
				
				for(Word word : tree.yieldWords())
				{
					stringBuilder.append(stringBuilder.length() > 0 ? " " : "");
					stringBuilder.append(word.word());
				}
				
				//System.out.println(stringBuilder.toString());
				
				for(List<HasWord> clause : new DocumentPreprocessor(new StringReader(stringBuilder.toString())))
				{
					sentences.add(clause);
				}
			}
		}
		
		//System.out.println();
		return sentences;
	}
	
	private String parseText(String text)
	{
		List<String> sentences = new ArrayList<String>();
		String formatted;
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
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
		String commandString = "command";
		
		if(jsonObject.containsKey(commandString))
		{
			String command = (String) jsonObject.get(commandString);
			
			if(command.equals("init"))
			{
				semgrexPatterns = new HashMap<SemgrexPattern, String>();
			}
			else if(command.equals("split_sentences"))
			{
				setSplitSentences(Boolean.parseBoolean((String) jsonObject.get("value")));
			}
			else if(command.equals("parse"))
			{
				return parseText((String) jsonObject.get("text"));
			}
			else if(command.equals("add_pattern"))
			{
				semgrexPatterns.put(SemgrexPattern.compile((String) jsonObject.get("pattern")), (String) jsonObject.get("class"));
			}
			else if(command.equals("classify"))
			{
				return classifyText((String) jsonObject.get("text"));
			}
			else if(command.equals("end"))
			{
				return shutdownCommand;
			}
		}
		
		return null;
	}
	
	public void setSplitSentences(boolean splitSentences)
	{
		this.splitSentences = splitSentences;
	}
	
	public static void main(String[] args) throws IOException, ParseException
	{
		int port = 9000;
		SemgrexClassifierHelper semgrexClassifierHelper = new SemgrexClassifierHelper();
		ServerSocket serverSocket = new ServerSocket(port);
		
		System.out.println("Listening on port " + port + ".");
		
		while(true)
		{
			Socket clientSocket = serverSocket.accept();
		    PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
		    BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
		    String inputLine, outputLine;
		    
		    while((inputLine = in.readLine()) != null)
		    {
		    	outputLine = semgrexClassifierHelper.receiveCommand(inputLine);
		    	
		    	if(outputLine != null)
		    	{
		    		if(outputLine.equals(shutdownCommand))
		    		{
		    			clientSocket.close();
		    			serverSocket.close();
		    			return;
		    		}
		    		
		    		out.println(outputLine.replace("\n", "__NEWLINE__"));
		    	}
		    }
		    
		    clientSocket.close();
		}
	}
}
