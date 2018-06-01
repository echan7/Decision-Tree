import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl{
	private DecTreeNode root;
	//ordered list of attributes
	private List<String> mTrainAttributes; 
	//
	private ArrayList<ArrayList<Double>> mTrainDataSet;
	//Min number of instances per leaf.
	private int minLeafNumber = 10;
	private int major;
	private int singleLabel;
	private ArrayList<DataBinder> sortedSet;
	private ArrayList<Double> bestSplitPointList;
	private ArrayList<Double> splitThreshold;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(ArrayList<ArrayList<Double>> trainDataSet, ArrayList<String> trainAttributeNames, int minLeafNumber) {
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = trainDataSet;
		this.minLeafNumber = minLeafNumber;
		this.root = buildTree(this.mTrainDataSet);
	}
	
	public DecTreeNode getRoot(){
		return this.root;
	}
	
	private boolean singleLabel(ArrayList<ArrayList<Double>> dataSet){
		Double s = null;
		for(ArrayList<Double> meow: dataSet){
			if(s == null){
				s = meow.get(meow.size()-1);             
			}else {
				if(!(meow.get(meow.size()-1).equals(s))){
					return false;
				}
			}
		}
		singleLabel = s.intValue();
		return true;
		
	}
	
	private DecTreeNode buildTree(ArrayList<ArrayList<Double>> dataSet){
		// TODO: add code here
		sortedSet = new ArrayList<DataBinder>();
		for(int i =0; i < dataSet.size(); i++){
			sortedSet.add(new DataBinder(dataSet.get(i).size()-1, dataSet.get(i)));
		}
		if(dataSet.isEmpty()){
			return new DecTreeNode(1, "", 0.0);	
		}else if(singleLabel(dataSet)){
			return new DecTreeNode(singleLabel, "", 0.0);
		}else if(dataSet.size() <= minLeafNumber){
			majority(sortedSet, 0, sortedSet.size()-1);
			return new DecTreeNode(major, null, 0.0);
		}
		
		ArrayList<String> names = new ArrayList<String>();
		for(String e : mTrainAttributes){
			names.add(e);
		}
		rootInfoGain(dataSet, names, minLeafNumber, 1);
		Double bestInfoGain = bestSplitPointList.get(0);
		int indexOfBest = 0;
		for(int i = 1; i < bestSplitPointList.size(); i++){
			if(bestSplitPointList.get(i) > bestInfoGain){
				bestInfoGain = bestSplitPointList.get(i);
				indexOfBest  = i;
			}else if(bestSplitPointList.get(i).equals(bestInfoGain)){
					indexOfBest = i;
			}
		}
		
		DecTreeNode newNode = new DecTreeNode(major, names.get(indexOfBest), splitThreshold.get(indexOfBest));
		ArrayList<ArrayList<Double>> lessThan = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> moreThan = new ArrayList<ArrayList<Double>>();
		for(ArrayList<Double> e: dataSet){
			if(e.get(indexOfBest) <= newNode.threshold){
				lessThan.add(e);
			}else{
				moreThan.add(e);
			}
		}
		
		newNode.left = buildTree(lessThan);
		newNode.right = buildTree(moreThan);
		
		return newNode;
		
	}	
	
	public int classify(List<Double> instance, DecTreeNode root) {
		// TODO: add code here
		if(root.isLeaf()){
			return root.classLabel;
		}
		
		int indexOfAttribute = 0;
		for(int i = 0; i < mTrainAttributes.size(); i++){
			if(mTrainAttributes.get(i).equals(root.attribute)){
				indexOfAttribute = i;
				break;
			}
		}
		
		if(instance.get(indexOfAttribute) <= root.threshold){
			return classify(instance, root.left);
		}else{
			return classify(instance, root.right);
		}

	}
	
	public void rootInfoGain(ArrayList<ArrayList<Double>> dataSet, ArrayList<String> trainAttributeNames, int minLeafNumber, int mode) {
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = dataSet;
		this.minLeafNumber = minLeafNumber;
		// TODO: add code here
		bestSplitPointList = new ArrayList<Double>();
		splitThreshold = new ArrayList<Double>();
		for(int j = 0; j < mTrainAttributes.size(); j++){
			sortedSet = new ArrayList<DataBinder>();
			for(int i = 0; i < mTrainDataSet.size(); i++){
				sortedSet.add(new DataBinder(j, mTrainDataSet.get(i)));
			}
			Collections.sort(sortedSet);	
			
			
			rearmDB(mTrainDataSet.get(0).size()-1, sortedSet);
			ArrayList<Double> splitPoints = new ArrayList<Double>();
			for(int i = 0; i < sortedSet.size(); i++){
				if((i+1) == sortedSet.size()){
					break;
				}
				DataBinder curr = sortedSet.get(i);
				DataBinder curr2 = sortedSet.get(i+1);
				if(curr.getArgItem() != (curr2.getArgItem())){
					rearmDB(j, sortedSet);
					Double val = (curr.getArgItem() + curr2.getArgItem())/2;
					splitPoints.add(val);
				}
				rearmDB(mTrainDataSet.get(0).size()-1, sortedSet);
			}
			
			rearmDB(j, sortedSet);
			
			int indexOfLessThan = 0;
			int sizeOfLessThan = 0;
			ArrayList<Double> splitPointsInfoGain = new ArrayList<Double>();
			for(int i =0; i < splitPoints.size(); i++){
				double threshold = splitPoints.get(i);
				for(int k =  0; k < sortedSet.size(); k++){
					if(sortedSet.get(k).getArgItem() > threshold){
						indexOfLessThan = k-1;
						sizeOfLessThan = k;
						break;
					}
					
					if(k+1 == sortedSet.size()){
						indexOfLessThan = k;
						sizeOfLessThan = sortedSet.size();
					}
				}
				
				rearmDB(mTrainDataSet.get(0).size()-1, sortedSet);
				
				double lessThan = 0.0;
				if(sizeOfLessThan != 0){
					int majority = majority(sortedSet,0, indexOfLessThan);
					int minority = sizeOfLessThan - majority;
					if(majority != 0 && minority != 0){
						lessThan = (double)sizeOfLessThan/sortedSet.size() * entropy((double)majority, (double)minority, (double)sizeOfLessThan);
					}	
				}
				int indexOfMoreThan = indexOfLessThan + 1;
				int sizeOfMoreThan = sortedSet.size() - sizeOfLessThan;
				double moreThan = 0.0;
				if(sizeOfMoreThan != 0){
					int majority = majority(sortedSet, indexOfMoreThan, sortedSet.size()-1);
					int minority = sizeOfMoreThan - majority;
					if(majority != 0 && minority != 0){
						moreThan = (double)sizeOfMoreThan/sortedSet.size() * entropy((double)majority, (double)minority, (double)sizeOfMoreThan);
					}
				}
				
				
				int mjor = majority(sortedSet, 0, sortedSet.size()-1);
				int mior = sortedSet.size() - mjor;
				double classInfo =  entropy((double)mjor, (double)mior, (double)sortedSet.size());;
				
				double infoGain = classInfo - (lessThan + moreThan);
				splitPointsInfoGain.add(infoGain);
				
				rearmDB(j, sortedSet);
				
			}
			
			Double highest = splitPointsInfoGain.get(0);
			int indexOfHighest = 0;
			for(int i = 1; i < splitPointsInfoGain.size(); i++){
				if(splitPointsInfoGain.get(i) > highest){
					highest = splitPointsInfoGain.get(i);
					indexOfHighest = i;
				}else if(splitPointsInfoGain.get(i) == highest){
					if(splitPoints.get(i) > splitPoints.get(indexOfHighest)){
						indexOfHighest = i;
					}
				}
			}
			splitThreshold.add(splitPoints.get(indexOfHighest));
			bestSplitPointList.add(splitPointsInfoGain.get(indexOfHighest));
			
		}	
			
		
		//TODO: modify this example print statement to work with your code to output attribute names and info gain. Note the %.6f output format.
		if(mode == 0){
			for(int i = 0; i<bestSplitPointList.size(); i++){
				System.out.println(this.mTrainAttributes.get(i) + " " + String.format("%.6f", bestSplitPointList.get(i)));
			}	
		}
	}
	
	private void rearmDB(int meow, ArrayList<DataBinder> sortedSet){
		for(int i =0; i < sortedSet.size(); i++){
			sortedSet.get(i).i = meow;
		}
	}
	
	private int majority(ArrayList<DataBinder> meow, int first, int last){
		int countZero = 0;
		int countOne = 0;
		for(int i = first; i <= last; i++){
			if(meow.get(i).getArgItem() == 1){
				countOne++;
			}else if(meow.get(i).getArgItem() == 0){
				countZero++;
			}
		}
		
		if(countZero > countOne){
			major = 0;
			return countZero;
		}else{
			major = 1;
			return countOne;
		}
	}

	
	private double entropy(double a, double b, double c){
		double p = (double)a/c;
		double q = (double)b/c;
		double infoGain = - (p * Math.log(p)/Math.log(2.0)  + q * Math.log(q)/Math.log(2.0));
		return infoGain;
	}
	
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode("", this.root);
	}

	/**
	 * Recursively prints the tree structure, left subtree first, then right subtree.
	 */
	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + node.attribute;
			
		System.out.print(printStr + " <= " + String.format("%.6f", node.threshold));
		if(node.left.isLeaf()){
			System.out.println(": " + String.valueOf(node.left.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%.6f", node.threshold));
		if(node.right.isLeaf()){
			System.out.println(": " + String.valueOf(node.right.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
		
		
	}
	
	public double printAccuracy(int numEqual, int numTotal){
		double accuracy = numEqual/(double)numTotal;
		System.out.println(accuracy);
		return accuracy;
	}

	/**
	 * Private class to facilitate instance sorting by argument position since java doesn't like passing variables to comparators through
	 * nested variable scopes.
	 * */
	private class DataBinder implements Comparable<DataBinder>{
		
		public ArrayList<Double> mData;
		public int i;
		public DataBinder(int i, ArrayList<Double> mData){
			this.mData = mData;
			this.i = i;
		}
		public double getArgItem(){
			return mData.get(i);
		}
		public ArrayList<Double> getData(){
			return mData;
		}
		@Override
		public int compareTo(DataBinder meow) {
			// TODO Auto-generated method stub
			if(getArgItem() < meow.getArgItem()){
				return -1;
			}else if(getArgItem() > meow.getArgItem()){
				return 1;
			}else{
				if(mData.get(mData.size()-1) > meow.getData().get(meow.getData().size()-1)){
					return 1;
				}else if(mData.get(mData.size()-1) < meow.getData().get(meow.getData().size()-1)){
					return -1;
				}
				return 0;
			}
			
		}
		
	}

}
