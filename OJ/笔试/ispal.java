import java.util.Scanner;
import String;
import Integer;

public class ispal{
    public static void main(String[] args){
           //其他代码
           Scanner sc = new Scanner(System.in);
           while(sc.hasNextLine()) { 
            int T= sc.nextInt(); //读下一个整型字符串
            for(int i = 0; i<T;i++){
                int n = sc.nextInt();
                int b_n = d2b(n);
                int r_b_n = pal(n);
                if (b_n == r_b_n){
                    System.out.println("YES");
                }
                else{
                    System.out.println("NO");
                }
            }
        }
        sc.close(); //用完后关闭扫描器是一个好的习惯
    }

    public static int d2b(int num){
        if (num < 2){
            return num;
        }
        int ret = 0;
        int c = 1;
        while(num > 0){
            ret += c * (num % 2);
            num = num / 2;
            c *= 10;
        }
        return ret;
    }

    public static int pal(int num){
        int r_num = 0;
        int c = 1;
        while(num > 0){
            r_num += c * (num % 10);
            num = num / 10;
            c *= 10;
        }
        return r_num;
    }

}
