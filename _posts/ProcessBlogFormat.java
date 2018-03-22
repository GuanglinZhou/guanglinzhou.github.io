

import java.io.*;
import java.util.Scanner;

/**
 * @author guanglinzhou
 * @link https://github.com/GuanglinZhou
 * @date 2018/03/22
 */
public class ProcessBlogFormat {
    public static void main(String[] args) {

        ProcessBlogFormat processBlogFormat = new ProcessBlogFormat();
        Scanner scanner = new Scanner(System.in);
        System.out.println("请输入路径：");
        processBlogFormat.processBlog(scanner.next());
    }

    public void processBlog(String path) {
        File fileRead = new File(path);
        File fileWrite = new File("wt_" + path);
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(fileRead));
            try {
                BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(fileWrite));
                String s = null;
                while ((s = bufferedReader.readLine()) != null) {
                    if (s.length() == 0) {
                        bufferedWriter.newLine();
                    } else {
                        for (int i = 0; i < s.length() - 1; i++) {
                            if (s.charAt(i) == '$' && s.charAt(i + 1) != '$') {
                                bufferedWriter.write("$$");
                            } else if (s.charAt(i) == '$' && s.charAt(i + 1) == '$') {
                                bufferedWriter.write('$');
                            } else {
                                bufferedWriter.write(s.charAt(i));
                            }
                        }
                        bufferedWriter.write(s.charAt(s.length() - 1));
                        bufferedWriter.newLine();

                    }

                }
                bufferedReader.close();
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            //删除将新文件替换旧文件,文件重名直接替换，不用语句删除旧文件
//            fileRead.delete();
            fileWrite.renameTo(new File(path));
            System.out.println("文件格式调整完毕，文件名为：" + path);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}

