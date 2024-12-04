package ir.vsr;

import java.io.*;
import java.util.*;
import java.lang.*;

import ir.utilities.*;
import ir.classifiers.*;

/**
 *  Simple system for testing retrieval based on a deep-learned embeddings of a corpus of documents 
 *  and queries that have already been encoded. Assumes that document and queries are files that contain
 *  precomputed deep embeddings rather than the original text.  This retriever simply computes similarity of
 *  an embedded query to embeddings of all the documents (using either Euclidian distance or cosine 
 *  similarity) to rank retrievals. 
 * @author Ray Mooney
 */
public class DeepRetriever {
  
    /**
     * A list of all documents as DeepDocumentReference's.
     */
    public List<DeepDocumentReference> docRefs = null;

    /**
     * The dimensionality (size) of the deep document embeddings stored in files
     */
    public int dimension;

    /**
     * The directory from which the documents come. This directory should contain a file for
     * each document that contains a dense vector (list of real-values separated by spaces)
     * of length 'dimension' that give the deep embedding for this document
     */
    public File dirFile = null;

    /**
     * Flag to indicate the use of cosine similarity when ranking retrievals,
     * default is to use inverse Euclidian distance as the similarity metric
     */
    public boolean useCosine = false;

  
  /**
   * Create a retriever for the document dense vectors stored in a directory.
   *
   * @param dirFile   The directory of files to read and store their deep vectors
                      Each file should contain a space-separated list of real values
   * @param useCosine Flag to indicate use of cosine rather than Euclidian distance for ranking
   */
    public DeepRetriever(File dirFile, boolean useCosine) {
	this.dirFile = dirFile;
	this.useCosine = useCosine;
	docRefs = new ArrayList<DeepDocumentReference>();
	readDocuments();
    }


    /**
     * Load document vectors from document files in dirFile.
     */
    protected void readDocuments() {
	File[] files = dirFile.listFiles();
	// Get dimension of ebeddings from the size of the vector in the first file
	dimension =  vectorDimension(files[0]);
	for (int k = 0; k < files.length; k++) {
	    System.out.print(files[k].getName() + ",");
	    docRefs.add(new DeepDocumentReference(files[k], dimension));
	}
    }


    /**
     * Determine the dimension of the vector stored in a file by counting the 
     * number of doubles it contains.
     */
    public static int vectorDimension(File file){
	int pos = 0;
	try {
	    // create scanner to read file
	    Scanner sc = new Scanner(file);
	    // read doubles into vector until EOF
	    while (sc.hasNext()) {
		sc.nextDouble();		
		pos++;
	    }
	}
	catch (IOException e) {
	    System.out.println("\nCould not load file: " + file);
	    System.exit(1);
	}
	return pos;
    }


    /**
     * Print out the document vectors stored in this retriever
     */
    protected void print() {
	System.out.println("\nNumber of files: " + docRefs.size());
	for (DeepDocumentReference docRef : docRefs) {
	    System.out.println(docRef.file.getName() +
			       " Vector: " + Arrays.toString(docRef.vector) +
			       "\n  Length:" + docRef.length);
	}
    }

    /**
     * Perform ranked retrieval on an input query encoded as a DeepDocumentReference.
     */
    public Retrieval[] retrieve(DeepDocumentReference queryDocRef) {
	// Make an array to store the final ranked Retrievals.
	Retrieval[] retrievals = new Retrieval[docRefs.size()];
	int pos = 0;
	double score;
	// Score each stored document using Euclidian or cosine to rank documents
	for (DeepDocumentReference docRef : docRefs) {
	    if (useCosine) 
		score = cosineSimilarity(queryDocRef, docRef);
	    else 
		score = 1/euclidianDistance(queryDocRef.vector, docRef.vector);
	    retrievals[pos] = new Retrieval(docRef, score);
	    pos++;
	}
	// Sort the retrievals based on their computed scores
	Arrays.sort(retrievals);
	return retrievals;
    }

    /**
     * Compute Euclidian distance between two vectors
     */
    public static double euclidianDistance(double[] array1, double[] array2)
    {
        double Sum = 0.0;
        for(int i=0; i<array1.length; i++) {
	    Sum += Math.pow((array1[i]-array2[i]),2.0);
        }
        return Math.sqrt(Sum);
    }

    /**
     * Compute cosine similiarity of two deep embedded documents
     */
    public static double cosineSimilarity(DeepDocumentReference doc1, DeepDocumentReference doc2)
    {
	double[] array1 = doc1.vector;
	double[] array2 = doc2.vector;
        double dotProduct = 0.0;
        for(int i=0; i<array1.length; i++) 
	    dotProduct += array1[i]* array2[i];
	return dotProduct / (doc1.length * doc2.length);
    }

  /**
   * Intended for a simple test run on a small corpus and optional test query file.
   * Load a directory of files and then print them, can also include an extra
   * test file as a 2nd file arg if you want to also test a sample query vector stored in a file,
   * in which case, you can include a "-cosine" flag to use cosine similarity instead of Euclidian
   * distance to rank retrievals.
   */
  public static void main(String[] args) {
      String dirName;
      if (args.length == 1)
	  dirName = args[0];
      else
	  dirName = args[args.length - 2];
      // Create a DeepRetriever for the files in the given directory.
      DeepRetriever retriever = new DeepRetriever(new File(dirName), args[0].equals("-cosine"));
      retriever.print();
      // See if there is an extra arg for a test query file
      if (args.length > 1) {
	  File file = new File(args[args.length - 1]);
	  DeepDocumentReference queryDocRef = new DeepDocumentReference(file, retriever.dimension);
	  // Run retrieval on the test query and print query vector and results
	  Retrieval[] retrievals = retriever.retrieve(queryDocRef);
	  System.out.println("\nQuery: " + Arrays.toString(queryDocRef.vector) + "\nRetrievals: ");
	  for(int i=0; i < retrievals.length; i++)
	      System.out.println(retrievals[i].docRef.file.getName() + " Score:" +
				 retrievals[i].score);
    }
  }

}

   
