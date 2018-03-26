

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
        StringBuilder stringBuilder = new StringBuilder(path);
        int initLength = path.length();
        stringBuilder.insert(initLength - 3, "_github");
        String pathNew = stringBuilder.toString();
        File fileWrite = new File(pathNew);
        //用于判断$$符号是在前面还是后面
        int index = 0;
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
                                index++;
                                if (index % 2 == 1) {
                                    bufferedWriter.write("\n");
                                } else {
                                    bufferedWriter.write("");
                                    i += 1;
                                }
                            } else {
                                bufferedWriter.write(s.charAt(i));
                            }
                        }
                        if (s.charAt(s.length() - 1) == '$') {
                            bufferedWriter.write("$$");
                        } else {
                            bufferedWriter.write(s.charAt(s.length() - 1));
                        }
                        bufferedWriter.newLine();

                    }

                }
                bufferedReader.close();
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            fileWrite.renameTo(new File(pathNew));
            System.out.println("文件格式调整完毕，文件名为：" + pathNew);
            System.out.println();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}


///Users/guanglinzhou/Documents/GitHub/guanglinzhou.github.io/_posts/2018-03-25-逻辑回归更新篇.md