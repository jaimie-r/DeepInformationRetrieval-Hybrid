package ir.webutils;

import java.io.*;
import java.net.*;

/**
 * Link is a class that contains a URL.  Subclasses of link may keep
 * additional information (such as anchor text & other attributes)
 *
 * @author Ted Wild and Ray Mooney
 */

public class Link {

  private URL url = null;

  /**
   * May be subclassed.  This constructor should not be invoked by
   * clients of <code>Link</code>.
   */
  protected Link() {
    url = null;
  }

  /**
   * Constructs a link with specified URL.
   *
   * @param url The URL for this link.
   */
  public Link(URL url) {
    this.url = url;
  }

  /**
   * Construct a link with specified URL string
   */
  public Link(String urlName) {
    try {
      this.url = cleanURL(new URL(urlName));
    }
    catch (MalformedURLException e) {
      System.err.println("Bad URL: " + urlName);
    }
  }

  /**
   * Returns the URL of this link.
   *
   * @return The URL of this link.
   */
  public final URL getURL() {
    return url;
  }

  public String toString() {
    return url.toString();
  }

  public boolean equals(Object o) {
    return (o instanceof Link) && ((Link) o).url.equals(this.url);
  }

  public int hashCode() {
    return url.hashCode();
  }

  public void cleanURL() {
      url = cleanURL(url);
  }

  /**
   * Standardize URL by using the URL connection to get a redirected URL
   *
   * @param url The unnormalized URL
   * @return a cleaned, normalized URL 
   */
  public static URL cleanURL(URL url) {
      if (!url.toString().startsWith("mailto:"))
	  try {
	      // System.out.println( "Cleaning URL: " + url);
	      URLConnection con = url.openConnection();
	      con.connect();
	      InputStream is = con.getInputStream();
	      URL redirectedURL = con.getURL();
	      // System.out.println( "Cleaned URL: " + redirectedURL);
	      is.close();    
	      URL refRemoved = removeRef(redirectedURL);
        URL usersMarkRemoved = removeUsersMark(refRemoved);
        return protocolCheck(usersMarkRemoved);
	  } catch (IOException e) {
	      return url;
	  }
      else return url;
  }

  /**
   * Remove the internal "ref" pointer in a URL if there is
   * one. This not part of the URL to a page itself
   */
  public static URL removeRef(URL url) {
    String ref = url.getRef();
    if (ref == null || ref.equals(""))
      return url;
    String urlName = url.toString();
    int pos = urlName.lastIndexOf("#");
    if (pos >= 0)
      try {
        return (new URL(urlName.substring(0, pos)));
      }
      catch (MalformedURLException e) {
        System.err.println("Bad Ref in URL: " + urlName);
      }
    return url;
  }

  /**
   * Written by Sumaya Al-Bedaiwi (salbedaiwi@utexas.edu)
   * Replace any links that explicitly go to the /users/ directory
   * with "~" to represent the user directly
   */
  public static URL removeUsersMark(URL url) {
    String urlName = url.toString();
    if (!urlName.contains("/users/")){
      return url;
    }
    int pos = urlName.indexOf("users/");
    if (pos >= 0){
      try {
        return (new URL(urlName.substring(0, pos) + "~" + urlName.substring(pos + ("users/".length()))));
      }
      catch (MalformedURLException e) {
        System.err.println("Bad Ref in URL: " + urlName);
      }
    }
    return url;
  
  }

  /**
   * Written by Sumaya Al-Bedaiwi (salbedaiwi@utexas.edu)
   * Attempts https protocol for all http protocol links.
   */
  public static URL protocolCheck(URL url) {
    if (url.getProtocol().equals("https")){
      return url;
    }
    String urlName = url.toString();
    int nonPos = urlName.indexOf("https");
    int pos = urlName.indexOf("http");
    if (nonPos == -1 && pos >= 0){
      try {
        return (new URL(urlName.substring(pos, ("http".length())) + "s" + urlName.substring(pos + ("http".length()))));
      }
      catch (MalformedURLException e) {
        return url;
      }
    }
    return url;
  
  }

  public static void main(String[] args) {
    System.out.println(new Link(args[0]));
  }

}

