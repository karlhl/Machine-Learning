### 个人Android 笔记

### UI布局

##### 按钮

###### Button

`Button btnLogin = (Button)findViewById(R.id.btnLogin);`

```
btnLogin.setOnClickListener(new View.OnClickListener() 
```



###### RadioButton

n选一按钮

```java
RadioButton rbtnLoginByUserName = (RadioButton)findViewById(R.id.rbtnLoginByUserName);
```



###### Toast

消息提示控件，显示临时提示内容。



###### AlertDialog

用于弹出让用户确认。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201030103216328.png" alt="image-20201030103216328" style="zoom:50%;" />

```java
AlertDialog alert = new AlertDialog.Builder(MainActivity.this).create();
alert.setTitle("用户协议确认");
alert.setIcon(R.drawable.asd);
alert.setMessage("注册新用户需要接受用户协议的约束，请认真查阅用户协议，并选择是否同意。");
alert.setButton(AlertDialog.BUTTON_POSITIVE, "确认", new DialogInterface.OnClickListener() {
    @Override
    public void onClick(DialogInterface dialog, int which) {
        View contentView = LayoutInflater.from(MainActivity.this).inflate(R.layout.popup_content,null,false);
        PopupWindow window = new PopupWindow(contentView,LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT,true);
        window.setTouchable(true);
        // 在下方进行弹出
        // window.showAsDropDown(btnReg,0,0,Gravity.BOTTOM);
        window.showAtLocation(getWindow().getDecorView(),Gravity.BOTTOM,0,0);
    }
});
alert.setButton(AlertDialog.BUTTON_NEGATIVE, "不同意", new DialogInterface.OnClickListener() {
    @Override
    public void onClick(DialogInterface dialog, int which) {
        ToastShow("只有接受用户协议，才能注册新用户。");
    }
});
alert.show();
```