package ucr.cs.dblab.textclassifier;

import java.util.Map;

import weka.core.Instances;

public class GoogleDatasetBalancer extends TextDatasetBalancer
{
	public static enum Parameter{SNIPPET_PARSING, SIMILARITY, DOC2VEC, THRESHOLD, QUOTES, SITE, KEYS, PROXIES, ENGINE_ID, PAGES_PER_QUERY};
	public static enum SnippetParsing{SENTENCE, ELLIPSIS, SNIPPET};
	public static enum Similarity{JACCARD, COSINE};
	private SnippetParsing snippetParsing;
	private Similarity similarity;
	private String[] keys, proxies;
	private String site, engineID;
	private int pagesPerQuery, keyIndex = 0;
	private double threshold;
	private boolean quotes;
	
	public GoogleDatasetBalancer(Map<Object, Object> parameters)
	{
		snippetParsing = parameters.containsKey(Parameter.SNIPPET_PARSING) ? (SnippetParsing) parameters.get(Parameter.SNIPPET_PARSING) : SnippetParsing.SENTENCE;
		similarity = parameters.containsKey(Parameter.SIMILARITY) ? (Similarity) parameters.get(Parameter.SIMILARITY) : Similarity.JACCARD;
		threshold = parameters.containsKey(Parameter.THRESHOLD) ? (Double) parameters.get(Parameter.THRESHOLD) : 1.0 / 3.0;
		quotes = parameters.containsKey(Parameter.QUOTES) ? (Boolean) parameters.get(Parameter.QUOTES) : false;
		site = parameters.containsKey(Parameter.SITE) ? (String) parameters.get(Parameter.SITE) : null;
		keys = (String[]) parameters.get(Parameter.KEYS);
		proxies = (String[]) parameters.get(Parameter.PROXIES);
		engineID = (String) parameters.get(Parameter.ENGINE_ID);
	}
	
	public Instances balance(Instances data)
	{
		return null;
	}
}