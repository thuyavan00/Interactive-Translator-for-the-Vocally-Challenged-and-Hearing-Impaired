import wx
import shutil
import os
def onButton(event):
    print("Button pressed.")
psxxo=wx.App()
frame=wx.Frame(None, -1, 'zip')
frame.SetDimensions(0,0,200,50)
# Create text input
dlg=wx.TextEntryDialog(frame, 'Enter the folder path','zip')
dlg.SetValue("Default")
cwd = os.getcwd()
if dlg.ShowModal()==wx.ID_OK:
    shutil.make_archive("new_dataset", 'zip', dlg.GetValue())
    wx.MessageBox("success", 'status', wx.OK)
    
dlg.Destroy()