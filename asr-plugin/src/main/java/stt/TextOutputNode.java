package stt;

import de.saar.coli.dialogos.marytts.plugin.Resources;
import com.clt.diamant.graph.nodes.AbstractOutputNode;
import com.clt.speech.SpeechException;
import com.clt.speech.tts.VoiceName;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TextOutputNode extends AbstractOutputNode {

    public enum PromptType implements IPromptType {
        text("Text"),
        expression("Expression"),
        groovy("GroovyScript");

        public IPromptType groovy() { return groovy; }
        public IPromptType expression() { return expression; }
        private String key;
        public IPromptType[] getValues() { return values(); };

        private PromptType(String key) {
            this.key = key;
        }

        @Override
        public String toString() {
            return Resources.getString(this.key);
        }
    }

    @Override
    public IPromptType getDefaultPromptType() {
        return PromptType.text;
    }

    @Override
    public List<VoiceName> getAvailableVoices() {
        return Collections.singletonList(new VoiceName("", null));
    }

    @Override
    public void speak(String prompt, Map<String, Object> properties) throws SpeechException {
        System.out.println("speech output: " + prompt);
    }

    @Override
    public String getResourceString(String key) {
        return key;
    }

    @Override
    public void stopSynthesis() {
        // nothing to be done for text output
        System.err.println("being asked to stop speaking");
    }

}