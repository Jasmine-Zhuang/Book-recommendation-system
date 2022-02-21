"""CSC111 Winter 2020 Project Phase 2

Instructions
===============================

This Python module contains the program of CSC111 Final Project.
You need to install tkinter to run this file.

Copyright and Usage Information
===============================

This file is provided solely for the final project for CSC111 at the University of
Toronto St. George campus. All forms of distribution of this code, whether as given
or with any changes, are expressly prohibited. For more information on copyright for
this file, please contact us.

This file is Copyright (c) 2021 Elaine Dai, Nuo Xu, Tommy Gong and Jasmine Zhuang.
"""
import tkinter as tk
from tkinter import ttk
from tkinter import *
from typing import Any
import helper as h

################################################################################
# global variables
################################################################################
BOOK_NAME = []
ALL_BOOK = []
LARGE_FONT = ('Verdana', 12)
PATH = ''
SELECTED_CLUSTER = []
SELECTED_BOOK_GRAPH = Any
CUSTOMIZED_PATH = ''
review_graph = h.load_weighted_review_graph('dataset.csv')
BOOK_GRAPH = h.create_book_graph(review_graph, threshold=0.01)
CLUSTER = [{book} for book in BOOK_GRAPH.get_all_vertices()]


################################################################################
# Window 1 (Collect inputs for <find_clusters>)
################################################################################
def click1() -> None:
    """Assign text to global variables PATH and NUM_CLUSTERS
    """
    global PATH
    global NUM_CLUSTERS
    PATH = path.get()
    NUM_CLUSTERS = num_cluster.get()


def find_clus() -> None:
    """calls the function find_clusters in helper.py by using global variables as input
    """
    global PATH
    global NUM_CLUSTERS
    h.find_clusters(BOOK_GRAPH, CLUSTER, int(NUM_CLUSTERS), PATH)


starting2 = Tk()
gif_photo = PhotoImage(file='book.gif')
Label(starting2, image=gif_photo, bg='mint cream').grid(row=0, column=0, sticky=E)
Label(starting2, text='\n Please enter a csv path to save result:', bg='dark slate gray',
      fg='white',
      font='none 12 bold').grid(row=2, column=0, sticky=W)
Label(starting2, text='\n Please enter the cluster number:', bg='dark slate gray', fg='white',
      font='none 12 bold').grid(row=4, column=0, sticky=W)
Label(starting2, text='\n ', bg='dark slate gray', fg='white',
      font='none 12 bold').grid(row=5, column=0, sticky=W)
Label(starting2, text="Reminder: All blanks in this page needs to be filled out if you want to "
                      "custom your csv.\n\n "
                      "1.If you don't want to generate a csv file, please click the button (Skip "
                      "this page). \n"
                      "2. If you want to generate a csv file:"
                      "enter a path to save the generated csv file\n"
                      "and enter the cluster number(an integer less than the total number of "
                      "books). ",
      bg='mint cream',
      fg='gray16', font='none 12 bold').grid(row=11, column=0, sticky=W)
starting2.configure(background='dark slate gray')

path = Entry(starting2, width=20, bg='white')
path.grid(row=3, column=0, sticky=W)
num_cluster = Entry(starting2, width=20, bg='white')
num_cluster.grid(row=5, column=0, sticky=W)
Button(starting2, text='Save entries', width=10, command=click1).grid(row=7, column=0, sticky=W)
Button(starting2, text='Create csv', width=10,
       command=lambda: find_clus()).grid(row=8, column=0, sticky=W)
starting2.title("Book Recommendation System")
Button(starting2, text='Skip this page', width=10,
       command=lambda: starting2.destroy()).grid(row=9,
                                                 column=0,
                                                 sticky=W)
Label(starting2, text='\n', bg='dark slate gray', fg='white', font='none 12 bold').grid(row=10,
                                                                                        column=0,
                                                                                        sticky=W)
starting2.mainloop()


################################################################################
# Window 2 (Collect inputs for <find_all_recommended_books>)
################################################################################


def click() -> None:
    """Assign text to global variable BOOK_NAME and CUSTOMIZED_PATH, then destroy this window
    """
    global BOOK_NAME
    global CUSTOMIZED_PATH
    CUSTOMIZED_PATH = customized_path.get()
    BOOK_NAME = text.get()
    starting.destroy()


starting = Tk()
gif_photo = PhotoImage(file='book.gif')
Label(starting, image=gif_photo, bg='mint cream').grid(row=0, column=0, sticky=E)
Label(starting, text='\n Enter favorite book:', bg='dark slate gray', fg='white',
      font='none 12 bold').grid(row=2, column=0, sticky=W)
Label(starting, text='\n Enter customized path(else leave blank):', bg='dark slate gray',
      fg='white',
      font='none 12 bold').grid(row=4, column=0, sticky=W)
Label(starting, text="\n Instructions: \n\n1. Enter favorite book \n only book id is allowed "
                     "\n for example: \n     book849\n\n2. Enter customized path(optional)\nthe "
                     "path should "
                     "be entered as a string \n for example:  \n 'clusters_20.csv, "
                     "clusters_50.csv'\n \nLeave it "
                     "blank will use our default data ", bg='mint cream', fg='gray16',
      font='none 12 bold'). \
    grid(row=6, column=0, sticky=E)
starting.configure(background='dark slate gray')
text = Entry(starting, width=20, bg='white')
text.grid(row=3, column=0, sticky=W)

customized_path = Entry(starting, width=20, bg='white')
customized_path.grid(row=5, column=0, sticky=W)
Button(starting, text='Submit', width=10, command=click).grid(row=7, column=0, sticky=W)
Label(starting, text='\n', bg='dark slate gray', fg='white', font='none 12 bold').grid(row=7,
                                                                                       column=0,
                                                                                       sticky=W)
starting.title("Book Recommendation System")
starting.mainloop()


################################################################################
# Recommend System
################################################################################
class RecommendSystem(tk.Tk):
    """This class manages the frame structure
    """

    def __init__(self, *args: Any, **kwargs: Any):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Book Recommendation System")
        window = tk.Frame(self)
        window.pack(side='left', fill='both')
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for each_frame in (RecommendPage, DoubleConfirm):
            frame = each_frame(window, self)
            self.frames[each_frame] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame(RecommendPage)

    def show_frame(self, target: Any) -> None:
        """The function raise the target frame
        """
        frame = self.frames[target]
        frame.tkraise()


class RecommendPage(tk.Frame):
    """This page is the main page for book recommendation system
    """

    def __init__(self, parent, controller: Any) -> None:
        global BOOK_NAME
        global BOOK_GRAPH
        global ALL_BOOK
        global CUSTOMIZED_PATH
        ALL_BOOK = h.find_all_recommended_books(BOOK_GRAPH, str(BOOK_NAME[:]), CUSTOMIZED_PATH)
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='recommend books ',
                         font=LARGE_FONT,
                         bg='dark slate gray', fg='white')
        label.pack(pady=10, padx=10)

        label1 = ttk.Label(self, text='Content', font=LARGE_FONT)
        label1.pack()

        output = tk.Text(self, width=75, height=3, background='black', fg='white')
        output.pack()

        def replace1(t: tk.Text, books_list: list[str], num: int) -> None:
            """The function clears the textbox and replace the original 5 books with 5
            recommended books, the book which deleted by user will be replaced by a new
            recommend book
            """
            global ALL_BOOK
            t.delete(0.0, END)
            if len(books_list) < 5:
                new_list = ["All books had been recommended"]
                t.insert(END, new_list)
                pass

            new_list = books_list[:6]
            new_list.pop(num)
            assert len(new_list) == 5
            t.insert(END, new_list)
            ALL_BOOK.pop(num)

        label2 = tk.Label(self, text='Click any unlike book(from left to right)',
                          font=LARGE_FONT,
                          bg='dark slate gray', fg='white')
        label2.pack(pady=10, padx=10)

        delete_1_button = ttk.Button(self, text="1",
                                     command=lambda: replace1(output, ALL_BOOK, 0))
        delete_1_button.pack()
        print(ALL_BOOK[:10])

        delete_2_button = ttk.Button(self, text="2",
                                     command=lambda: replace1(output, ALL_BOOK, 1))
        delete_2_button.pack()

        delete_3_button = ttk.Button(self, text="3",
                                     command=lambda: replace1(output, ALL_BOOK, 2))
        delete_3_button.pack()

        delete_4_button = ttk.Button(self, text="4",
                                     command=lambda: replace1(output, ALL_BOOK, 3))
        delete_4_button.pack()

        delete_5_button = ttk.Button(self, text="5",
                                     command=lambda: replace1(output, ALL_BOOK, 4))
        delete_5_button.pack()

        clear_button = ttk.Button(self, text="clear result",
                                  command=lambda: output.delete(0.0, END))
        clear_button.pack()

        ttk.Button(self, width=0, text="",
                   command=output.insert(END, ALL_BOOK[:5]))

        label5 = tk.Label(self, text='Are you satisfied with all the recommended books? ',
                          font=LARGE_FONT,
                          bg='dark slate gray', fg='white')
        label5.pack()

        confirm_button = ttk.Button(self, text="Yes",
                                    command=lambda: controller.show_frame(DoubleConfirm))
        confirm_button.pack()


def quit_starting() -> None:
    """This function quit the DoubleConfirm page of GUI
    """
    com1 = tk.Button(starting2, text='Quit', command=quit())
    com1.pack(side=tk.BOTTOM)


class DoubleConfirm(tk.Frame):
    """The PageThree class which works as a popup window for user to double conform before exit to the StartPage
    """

    def __init__(self, parent, controller: Any) -> None:
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Are you sure you don’t need any new recommendation?',
                         font=LARGE_FONT,
                         bg='dark slate gray', fg='white')
        label.pack(pady=10, padx=10)
        space_1 = tk.Label(self, text='', bg='white')
        space_1.pack()
        space_0 = tk.Label(self, text='', bg='white')
        space_0.pack()
        back_button = ttk.Button(self, text="I prefer more recommend books",
                                 command=lambda: controller.show_frame(RecommendPage))
        back_button.pack()
        confirm_button = ttk.Button(self, text="Quit",
                                    command=lambda: quit_starting())
        confirm_button.pack()


app = RecommendSystem()
app.mainloop()
