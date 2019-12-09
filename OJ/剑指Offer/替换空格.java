public class Solution {
    // 使用StringBuffer 即可动态地构建
    public String replaceSpace_1(StringBuffer str) {
        if(str.length() == 0){
            return "";
        }
        StringBuffer ret = new StringBuffer("");
        for(int i=0;i<str.length();i++){
            char c = str.charAt(i);
            if(c == ' '){
                ret.append("%20");
            }
            else{
                ret.append(c);
            }
        }
        return ret.toString();
    }
    public String replaceSpace_2(StringBuffer str) {
        int spaceCount = 0;
        int l = str.length()-1;
        for(int i=0;i<=l;i++){
            if(str.charAt(i) == ' '){
                spaceCount++;
            }
        }
        int new_l = l + 2 * spaceCount;
        str.setLength(new_l+1);
        for(;l>=0;l--){
            if(str.charAt(l) == ' '){
                str.setCharAt(new_l--, '0');
                str.setCharAt(new_l--, '2');
                str.setCharAt(new_l--, '%');
            }
            else{
                str.setCharAt(new_l--, str.charAt(l));
            }
        }
        return str.toString();
    }
}