package stt;

import com.clt.dialogos.plugin.PluginRuntime;
import com.clt.dialogos.plugin.PluginSettings;
import com.clt.diamant.IdMap;
import com.clt.diamant.graph.Node;
import com.clt.xml.XMLReader;
import com.clt.xml.XMLWriter;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;


public class Plugin implements com.clt.dialogos.plugin.Plugin {
    @Override
    public void initialize() {
        Node.registerNodeTypes(com.clt.speech.Resources.getResources().createLocalizedString("IONode")),
            Arrays.asList(TextInputNode.class, TextOutputNode.class));
    }

    @Override
    public String getID() {
        return "speech-to-text";
    }

    @Override
    public String getName() {
        return "DialogOS Speech-To-Text Plugin";
    }

    @Override
    public Icon getIcon() {
        return UIManager.getIcon("FileView.computerIcon");
    }

    @Override
    public String getVersion() {
        return "1.0"
    }

    @Override
    public PluginSettings createDefaultSettings() {
        return new PluginSettings() {
            @Override
            public void writeAttributes(XMLWriter out, IdMap uidMap){
                // nothing to be written
            }

            @Override
            public JComponent createEditor() {
                return new JLabel();
            }

            @Override
            protected PluginRuntime createRuntime(Component parent) {
                return () -> {
                    // no runtime
                };
            }
        };
    }
}