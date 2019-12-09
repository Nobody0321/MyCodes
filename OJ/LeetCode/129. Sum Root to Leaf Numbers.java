import org.omg.PortableInterceptor.NON_EXISTENT;

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int pathSum = 0;

    public int sumNumbers(TreeNode root) {
        if(root != null){
            PathSums(root, 0);
        }
        return pathSum;
    }

    private void PathSums(TreeNode node, int path){
        if(node != null){
            path += node.val;
        }
        if (node.left != null || node.right != null){
            if(node.left != null){
                PathSums(node.left, path);
            }
            if(node.right!=null){
                PathSums(node.right, path);
            }
        }
        else{
            pathSum += path;
        }
    }
}