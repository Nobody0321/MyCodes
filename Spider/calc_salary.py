import tkinter as tk

window = tk.Tk()
window.title('这点工资算什么算？')
window.geometry('450x450')

tk.Label(window, text="当月工作天数：").place(x=50, y= 150)
tk.Label(window, text="当月税后工资：").place(x=50, y= 190)

var_labor_days = tk.StringVar()

entry_labor_days = tk.Entry(window, textvariable=var_labor_days)
entry_labor_days.pack()
total_salary = tk.Text(window, width=30, height=2)
total_salary.place(x=160, y=150)

def calc_salary():
    days = int(var_labor_days.get())
    salary = (days * 300 - 800) * 0.8 + 800
    total_salary.insert(tk.END, str(salary))

btn_calc = tk.Button(window, text='计算', width=30, height=2, command=calc_salary)
btn_calc.place(x=170, y=230)

window.mainloop()
