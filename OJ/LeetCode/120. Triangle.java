class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        for(int x=triangle.size()-2;x>=0;x--){
            List<Integer> level = triangle.get(x);
            for(int y=0;y<=x;y++){
                level.set(y, triangle.get(x).get(y) + (triangle.get(x+1).get(y) < triangle.get(x+1).get(y+1) ? triangle.get(x+1).get(y) : triangle.get(x+1).get(y+1)));
            }
            triangle.set(x, level);
        }
        return triangle.get(0).get(0);
    }
}