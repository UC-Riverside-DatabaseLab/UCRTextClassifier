package ucr.cs.dblab.textclassifier;

import java.sql.SQLException;

import weka.core.Instances;

public abstract class TextDatasetBalancer
{
	public abstract Instances balance(Instances data) throws SQLException;
}
