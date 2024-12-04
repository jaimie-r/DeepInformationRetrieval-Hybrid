package ir.vsr;

import java.io.*;
import java.util.*;

/**
 * A simple data structure that stores info about a deep-learning embedded document
 * that extends DocumentReference to include an embedded vector representation.
 * Assumes an embeddings file simple contains a list of real values separated by spaces
 * giving the dense vector for the document
 *
 * @author Ray Mooney
 */

public class DeepDocumentReference extends DocumentReference {

  /**
   * The deep embbeding vector for the document
   */
    public double[] vector;

  /**
   * Create a new text document for the given file.
   *
   * @param file      The file to make a docRef to
   * @param dimension The size of the vector embedding in this file
   */
    public DeepDocumentReference(File file, int dimension) {
	super(file, 0);  // Create a DocumentReference
	// Read in the embedding vector for this file
	this.vector = new double[dimension];
	try {
	    // create scanner to read file
	    Scanner sc = new Scanner(file);
	    int pos = 0;
	    // read doubles into vector until EOF
	    while (sc.hasNext()) {
		this.vector[pos] = sc.nextDouble();
		pos++;
	    }
	}
	catch (IOException e) {
	    System.out.println("\nCould not load file: " + file);
	    System.exit(1);
	}
	// Set the length to the Euclidian length of the vector
	this.length = L2Norm(vector);
    }

    /**
     * Computes Euclidian length (L2 norm) of a vector
     */
    public static double L2Norm(double[] data) {
        double ans = 0.0;
        for (int k = 0; k < data.length; k++) {
            ans += data[k] * data[k];
        }
        return (Math.sqrt(ans));
    }
    
  /**
   * For testing, print the read-in dense vector for a given file and print it out
   * Command args are <FILE> <DIMENSION> where <FILE> is the name of the file with
   * a deep emdedding and <DIMENSION> is the size (length) of the vector stored in this file
   */
  public static void main(String[] args) throws IOException {
      File file = new File(args[0]);
      int dimension = Integer.parseInt(args[1]);
      DeepDocumentReference docRef = new DeepDocumentReference(file, dimension);
      System.out.println(docRef.file.getName() +
			 " Vector: " + Arrays.toString(docRef.vector) +
			 "\nLength:" + docRef.length);
  }
}

