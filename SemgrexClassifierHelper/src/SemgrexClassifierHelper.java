import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
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
	public static enum Mode{INIT, TRAIN, EVALUATE, CLASSIFY};
	public static final String addPatternCommand = "add_pattern";
	public static final String classifyCommand = "classify";
	public static final String endCommand = "end";
	public static final String parseCommand = "parse";
	public static final String setModeCommand = "set_mode";
	public static final String splitSentencesCommand = "split_sentences";
	public static final String testCommand = "test";
	private static final String shutdownToken = "__SHUTDOWN__";
	private static final String sbar = "SBAR";
	private final MaxentTagger tagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
	private final DependencyParser parser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
	private final LexicalizedParser constituencyParser = LexicalizedParser.loadModel();
	private final JSONParser jsonParser = new JSONParser();
	private final Map<Mode, Set<String>> validCommands = new HashMap<Mode, Set<String>>(Mode.values().length);
	private Map<String, Double> distribution = new HashMap<String, Double>();
	private List<SemgrexPatternWrapper> semgrexPatterns;
	private Mode mode = Mode.INIT;
	private boolean splitSentences = false;
	
	public SemgrexClassifierHelper()
	{
		validCommands.put(Mode.INIT, new HashSet<String>());
		validCommands.put(Mode.TRAIN, new HashSet<String>());
		validCommands.put(Mode.EVALUATE, new HashSet<String>());
		validCommands.put(Mode.CLASSIFY, new HashSet<String>());
		validCommands.get(Mode.INIT).add(endCommand);
		validCommands.get(Mode.INIT).add(setModeCommand);
		validCommands.get(Mode.INIT).add(splitSentencesCommand);
		validCommands.get(Mode.TRAIN).add(addPatternCommand);
		validCommands.get(Mode.TRAIN).add(endCommand);
		validCommands.get(Mode.TRAIN).add(parseCommand);
		validCommands.get(Mode.TRAIN).add(setModeCommand);
		validCommands.get(Mode.TRAIN).add(splitSentencesCommand);
		validCommands.get(Mode.EVALUATE).add(endCommand);
		validCommands.get(Mode.EVALUATE).add(setModeCommand);
		validCommands.get(Mode.EVALUATE).add(testCommand);
		validCommands.get(Mode.CLASSIFY).add(classifyCommand);
		validCommands.get(Mode.CLASSIFY).add(endCommand);
		validCommands.get(Mode.CLASSIFY).add(setModeCommand);
	}
	
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
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				if(semgrexPatternWrapper.find(semanticGraph))
				{
					if(!distribution.containsKey(semgrexPatternWrapper.getClassLabel()))
					{
						distribution.put(semgrexPatternWrapper.getClassLabel(), 0.0);
					}
					
					distribution.put(semgrexPatternWrapper.getClassLabel(), distribution.get(semgrexPatternWrapper.getClassLabel()) + 1.0);
					break;
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
		
		for(Tree child : root.children())
		{
			if(!child.value().equals(sbar) && !child.value().equals("S"))
			{
				queue.add(child);
			}
		}
		
		while(!queue.isEmpty())
		{
			root.removeChild(root.objectIndexOf(queue.remove()));
		}
		
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
		
		root.setValue("ROOT");
		
		tree = root.prune(predicate);
		
		if(tree != null)
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
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			for(Tree tree : parseClauses(constituencyParser.parse(sentence)))
			{
				stringBuilder.delete(0, stringBuilder.length());
				
				for(Word word : tree.yieldWords())
				{
					stringBuilder.append(stringBuilder.length() > 0 ? " " : "");
					stringBuilder.append(word.word());
				}
				
				for(List<HasWord> clause : new DocumentPreprocessor(new StringReader(stringBuilder.toString())))
				{
					sentences.add(clause);
				}
			}
		}
		
/*		if(sentences.size() > 1)
		{
			System.out.println("Original text: " + text);
			
			for(List<HasWord> sentence : sentences)
			{
				for(HasWord hasWord : sentence)
				{
					System.out.print(hasWord.word() + " ");
				}
				
				System.out.println();
			}
			
			System.out.println();
		}
		*/
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
			
			if(validCommands.get(mode).contains(command))
			{
				if(command.equals(addPatternCommand))
				{
					semgrexPatterns.add(new SemgrexPatternWrapper(SemgrexPattern.compile((String) jsonObject.get("pattern")), (String) jsonObject.get("class")));
				}
				else if(command.equals(classifyCommand))
				{
					return classifyText((String) jsonObject.get("text"));
				}
				else if(command.equals(endCommand))
				{
					return shutdownToken;
				}
				else if(command.equals(parseCommand))
				{
					return parseText((String) jsonObject.get("text"));
				}
				else if(command.equals(setModeCommand))
				{
					setMode((String) jsonObject.get("mode"));
				}
				else if(command.equals(splitSentencesCommand))
				{
					setSplitSentences(Boolean.parseBoolean((String) jsonObject.get("value")));
				}
				else if(command.equals(testCommand))
				{
					testSemgrexPatterns((String) jsonObject.get("text"), (String) jsonObject.get("class"));
				}
			}
		}
		
		return null;
	}
	
	public void setMode(Mode mode)
	{
		this.mode = mode;
		
		switch(mode)
		{
			case INIT:
				break;
			case TRAIN:
				semgrexPatterns = new ArrayList<SemgrexPatternWrapper>();
				break;
			case EVALUATE:
				break;
			case CLASSIFY:
				semgrexPatterns = new ArrayList<SemgrexPatternWrapper>(new HashSet<SemgrexPatternWrapper>(semgrexPatterns));
				
				Collections.sort(semgrexPatterns);
				break;
			default:
				return;
		}
		
		System.out.println("Mode set to " + mode.toString() + ".");
	}
	
	public void setMode(String modeValue)
	{
		if(modeValue.equals("train"))
		{
			setMode(Mode.TRAIN);
		}
		else if(modeValue.equals("evaluate"))
		{
			setMode(Mode.EVALUATE);
		}
		else if(modeValue.equals("classify"))
		{
			setMode(Mode.CLASSIFY);
		}
	}
	
	public void setSplitSentences(boolean splitSentences)
	{
		this.splitSentences = splitSentences;
	}
	
	private void testSemgrexPatterns(String text, String classLabel)
	{
		SemanticGraph semanticGraph;
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				semgrexPatternWrapper.test(semanticGraph, classLabel);
			}
		}
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
		    
		    System.out.println("Connected to " + clientSocket.getInetAddress().getHostAddress() + ".");
		    
		    try
		    {
			    while((inputLine = in.readLine()) != null)
			    {
			    	outputLine = semgrexClassifierHelper.receiveCommand(inputLine);
			    	
			    	if(outputLine != null)
			    	{
			    		if(outputLine.equals(shutdownToken))
			    		{
			    			clientSocket.close();
			    			serverSocket.close();
			    			return;
			    		}
			    		
			    		out.println(outputLine.replace("\n", "__NEWLINE__"));
			    	}
			    }
		    }
		    catch(SocketException socketException)
		    {
		    	System.out.println("Disconnected from client.");
		    }
		    
		    clientSocket.close();
		}
	}
	
	private static final Predicate<Tree> predicate = new Predicate<Tree>()
	{
		@Override
		public boolean test(Tree tree)
		{
			return !tree.value().equals(sbar);
		}
	};
	
	public class SemgrexPatternWrapper implements Comparable<SemgrexPatternWrapper>
	{
		private SemgrexPattern semgrexPattern;
		private String classLabel;
		private Map<String, Double> distribution = new HashMap<String, Double>();
		
		public SemgrexPatternWrapper(SemgrexPattern semgrexPattern, String classLabel)
		{
			this.semgrexPattern = semgrexPattern;
			this.classLabel = classLabel;
		}
		
		public boolean equals(SemgrexPatternWrapper semgrexPatternWrapper)
		{
			return semgrexPattern.pattern().equals(semgrexPatternWrapper.semgrexPattern.pattern()) && classLabel.equals(semgrexPatternWrapper.getClassLabel());
		}
		
		public boolean find(SemanticGraph semanticGraph)
		{
			return semgrexPattern.matcher(semanticGraph).find();
		}
		
		public double getAccuracy()
		{
			if(distribution.size() > 0)
			{
				double sum = 0.0;
				
				for(Double d : distribution.values())
				{
					sum += d;
				}
				
				if(sum > 0.0)
				{
					for(String label : distribution.keySet())
					{
						distribution.put(label, distribution.get(label) / sum);
					}
				}
				
				return distribution.containsKey(classLabel) ? distribution.get(classLabel) / sum : 0.0;
			}
			
			return 0.0;
		}
		
		public String getClassLabel()
		{
			return classLabel;
		}
		
		public void test(SemanticGraph semanticGraph, String classLabel)
		{
			if(find(semanticGraph))
			{
				if(!distribution.containsKey(classLabel))
				{
					distribution.put(classLabel, 0.0);
				}
				
				distribution.put(classLabel, distribution.get(classLabel) + 1.0);
			}
		}

		@Override
		public int compareTo(SemgrexPatternWrapper semgrexPatternWrapper)
		{
			double accuracyA = getAccuracy(), accuracyB = semgrexPatternWrapper.getAccuracy();
			return accuracyA < accuracyB ? 1 : accuracyA > accuracyB ? -1 : 0;
		}
	}
}
