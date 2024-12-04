package ir.eval;

import java.io.*;
import java.util.*;
import java.lang.*;

import ir.utilities.*;
import ir.vsr.*;

/**
 * Contains methods for running evaluation experiments for information
 * retrieval, specifically the generation of recall-precision curves
 * for a given test corpus of query/relevant-documents pairs.
 *
 * This is a version for evaluating deep-learning based retrieval where
 * the documents have been pre-embedded and the vectors stored in a
 * directory of files with the same names as the original documents but just
 * containing a dense vector for the document (a line of real-values separated
 * by spaces).  And queries are also pre-embedded and the vectors stored in 
 * a directory of files, named Q1...Qn, each containing the dense vector for 
 * the Qi'th query in the queryFile (see the specs for this file below).
 *
 * @author Ray Mooney
 */

public class HybridExperiment {

    /**
     * The standard recall levels for which we want to plot precision values
     */
    public static final double[] RECALL_LEVELS = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

        // directory whose files should be indexed
    public File dir = null;

    public File embedDir = null;

    public File queries = null;

    public File queryDir = null;

    public Double lambda = 0.0;

    public File outFile = null;

    public HybridRetriever retriever = null;

    ArrayList<ArrayList<RecallPrecisionPair>> rpResults =
        new ArrayList<ArrayList<RecallPrecisionPair>>();

    ArrayList<double[]> interpolatedPrecisions = new ArrayList<double[]>();

    double[] averagePrecisions = null;

    public int queryIndex;

    public HybridExperiment(File dir, File embedDir, File queries, File queryDir, double lambda, 
            File outFile) throws IOException {
        this.dir = dir;
        this.embedDir = embedDir;
        this.queries = queries;
        this.queryDir = queryDir;
        this.lambda = lambda;
        this.outFile = outFile;
        queryIndex = 0;
        this.retriever = new HybridRetriever(dir, embedDir, DocumentIterator.TYPE_TEXT, false, false, lambda);
    }


    /**
     * Process and evaluate all queries and generate recall-precision curve
     */
    public void makeRpCurve() throws IOException {
        processQueries();
        // Use rpResults generated to interpolate a precision values for
        // each standard recall level for each query and store results in
        // interpolatedPrecisions
        for (ArrayList<RecallPrecisionPair> rpPairs : rpResults) {
        interpolatedPrecisions.add(interpolatePrecision(rpPairs));
        }
        // Compute the average precision values
        averagePrecisions = MoreMath.averageVectors(interpolatedPrecisions);
        System.out.println("\nAverage Interpolated Precisions:");
        MoreMath.printVector(averagePrecisions);
        System.out.println("");
        // Write results to output file and Gnuplot file for graphing
        writeRpCurve();
        graphRpCurve();
    }

    /**
     * Process each query in the queryFile and store evaluated results
     * in rpResults
     */
    void processQueries() throws IOException {
        BufferedReader in = new BufferedReader(new FileReader(queries));
        File[] queryFiles = queryDir.listFiles();
        Arrays.sort(queryFiles, Comparator.comparing(File::getName));
        while (processQuery(in, queryFiles)) ;
        in.close();
        // System.out.println("\n" + rpResults);
    }

    /**
     * Process the next query read from the query file reader and evaluate
     * results compared to known relevant docs also read from the query file.
     *
     * @return true if query successfully read, else false if no more queries
     * in query file
     */
    boolean processQuery(BufferedReader in, File[] queryFiles) throws IOException {
        String query = in.readLine();   // get the query
        if (query == null) return false;  // return false if end of file
        System.out.println("\nQuery " + queryFiles[queryIndex].getName() + ": " + query);

        // Process the query and get the ranked retrievals
        // First get the query document embedding from the query doc for this query
        // Assumes the dimension of the query vector is the same as that of the documents
        // stored in the retriever.
        DeepDocumentReference queryDocRef = new DeepDocumentReference(queryFiles[queryIndex],
                                        retriever.deepRetriever.dimension);
        Retrieval[] retrievals = retriever.retrieve(queryDocRef, query);
        System.out.println("Returned " + retrievals.length + " documents.");

        // Read the known relevant docs from query file and parse them
        // into an ArrayList of String's of relevant file names.
        String line = in.readLine();
        ArrayList<String> correctRetrievals = MoreString.segment(line, ' ');
        System.out.println(correctRetrievals.size() + " truly relevant documents.");
        
        // Generate Recall/Precision points and save in rpResults
        rpResults.add(evalRetrievals(retrievals, correctRetrievals));

        // Read the blank line delimiter between queries in the query file
        line = in.readLine();
        if (!(line == null || line.trim().equals(""))) {
        System.out.println("\nCould not find blank line after query, bad queryFile format");
        System.exit(1);
        }
        queryIndex++;
        return true;
    }

    /**
     * Compare retrieved docs to relevant docs and compute recall/precision
     * points.  Goes down ranked retrievals in order, stopping at each
     * relevant document and computing a RecallPrecisionPair for thresholding
     * at that point.
     *
     * @return An ArrayList of RecallPrecisionPair's
     */
    ArrayList<RecallPrecisionPair> evalRetrievals(Retrieval[] retrievals,
                                                    ArrayList<String> correctRetrievals) {
        ArrayList<RecallPrecisionPair> rpList = new ArrayList<RecallPrecisionPair>();
        // Number of correctly retrieved docs at any given point
        double goodRetrievals = 0;
        // Examine each ranked retrieval in order to compute rp pairs
        for (int i = 0; i < retrievals.length; i++) {
        // Current number of retrievals considered
        int numRetrieved = i + 1;
        // Check if this retrieval is in the list of relevant docs
        if (correctRetrievals.contains(retrievals[i].docRef.file.getName())) {
            goodRetrievals++;  // This is a relevant retrieval
            // Compute recall and precision for first numRetrieved docs
            double recall = goodRetrievals / correctRetrievals.size();
            double precision = goodRetrievals / numRetrieved;
            System.out.println(MoreString.padToLeft(numRetrieved, 4) +
                " is relevant; Recall = " +
                MoreString.padToLeft(MoreMath.roundTo(100 * recall, 3) + "%", 7) +
                "; Precision = " +
                MoreString.padToLeft(MoreMath.roundTo(100 * precision, 3) + "%", 7));
            // Create a RecallPrecisionPair for this point and add to rpList
            rpList.add(new RecallPrecisionPair(recall, precision));
        }
        }
        return rpList;
    }

    /**
     * Interpolate precision values for each standard recall level
     * in RECALL_LEVELS from the list of rpPairs for a given query.
     * See textbook for details.
     */
    double[] interpolatePrecision(ArrayList<RecallPrecisionPair> rpPairs) {
        // Array of interpolated precisions
        double[] precisions = new double[RECALL_LEVELS.length];
        // Compute precision value for each recall level, starting
        // from the highest and working backwards
        for (int i = RECALL_LEVELS.length - 1; i >= 0; i--) {
        // Stores maximum precision for this recall level.
        // Interpolated precision at level i is the max
        // precision seen (or interpolated) at any recall
        // value between level i and level i+1, inclusive.
        double maxPrecision = 0.0;
        // Check each point in rpPairs to see if it is between
        // recall levels i and i+1, compute the max of these precision values.
        for (RecallPrecisionPair rpPair : rpPairs) {
            if (RECALL_LEVELS[i] <= rpPair.recall &&
                (i == RECALL_LEVELS.length - 1 ||  // no higher level i+1
                    rpPair.recall <= RECALL_LEVELS[i + 1])) {
            // If recall in correct interval, update max precision
            if (rpPair.precision > maxPrecision)
                maxPrecision = rpPair.precision;
            }
        }
        // Also consider the previously computed precision level for
        // the next highest recall level i+1, to include in max computation
        if (i != RECALL_LEVELS.length - 1 && precisions[i + 1] > maxPrecision)
            maxPrecision = precisions[i + 1];
        // Set precision at level i to be the proper max interpolated value
        precisions[i] = maxPrecision;
        }
        //  	System.out.print("\nInterpolated Precisions: ");
        //	MoreMath.printVector(precisions);
        // Return vector of final interpolated precisions
        return precisions;
    }

    /**
     * Write out the final interpolated recall/precision graph data.
     * One line for each recall/precision point in the form: 'R-value P-value'.
     * This is the format needed for GNUPLOT.
     */
    void writeRpCurve() throws IOException {
        PrintWriter out = new PrintWriter(new FileWriter(outFile));
        for (int i = 0; i < RECALL_LEVELS.length; i++)
        out.println(RECALL_LEVELS[i] + " " + averagePrecisions[i]);
        out.close();
    }

    /**
     * Write out an appropriate input file for GNUPLOT for the final recall
     * precision graph to the output file with a ".gplot" extension.
     * See GNUPLOT documentation.
     */
    void graphRpCurve() throws IOException {
        File graphFile = new File(outFile.getPath() + ".gplot");
        PrintWriter out = new PrintWriter(new FileWriter(graphFile));
        out.print("set xlabel \"Recall\"\nset ylabel \"Precision\"\n\nset terminal postscript color\nset size 0.75,0.75\n\nset style data linespoints\nset key top right\n\nset xrange [0:1]\nset yrange [0:1]\n\nplot \'" + outFile.getName() + "\' title \"VSR\"");
        out.close();
    }

    public static void main(String[] args) throws IOException {
        String dir = args[0];
        String embedDir = args[1];
        String queries = args[2];
        String queryDir = args[3];
        double lambda = Double.valueOf(args[4]);
        String outFile = args[5];

        HybridExperiment exper = new HybridExperiment(new File(dir), new File(embedDir), new 
            File(queries), new File(queryDir), lambda, new File(outFile));
        exper.makeRpCurve();
    }
}

