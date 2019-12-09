public class Solution {
    public boolean Find(int target, int [][] array) {
        // 思路： 从左下角开始，大了就向上，小了就向右

        int row=0;
            int col=array[0].length-1;
            while(row<=array.length-1&&col>=0){
                if(target==array[row][col])
                    return true;
                else if(target>array[row][col])
                    row++;
                else
                    col--;
            }
            return false;
    }
}